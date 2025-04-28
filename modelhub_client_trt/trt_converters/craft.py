"""
CRAFT TensorRT converter

Тест:
CUDA_VISIBLE_DEVICES=1 python3.12 -m modelhub_client_trt.trt_converters.craft

Приклад використання:
--------------------------------------------------------------------
converter = CraftTrtConverter()
converter.convert(
    original_model_path="craft.ts",
    engine_path="craft.trt",
    onnx_path="craft.onnx",
    model_config={"dynamic_h": (32, 256, 1280), "dynamic_w": (32, 256, 1280)},
    builder_config={"fp16_mode": True, "workspace_size_gb": 4},
)
--------------------------------------------------------------------
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Dict, Any, Optional

import torch
import onnx
import tensorrt as trt

from .base import BaseTrtConverter, TRT_LOGGER


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------


def _make_example_input(
        bsz: int,
        dynamic_h: tuple[int, int, int],
        dynamic_w: tuple[int, int, int],
        device: torch.device,
        fp16: bool,
) -> torch.Tensor:
    """Creates an example tensor using the OPT shape for dynamic dims."""
    _, opt_h, _ = dynamic_h
    _, opt_w, _ = dynamic_w
    dtype = torch.float16 if fp16 and device.type == "cuda" else torch.float32
    return torch.randn((bsz, 3, opt_h, opt_w), device=device, dtype=dtype)


# ----------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------


class CraftTrtConverter(BaseTrtConverter):
    """Конвертер CRAFT→ONNX→TensorRT10."""

    def convert(  # noqa: C901
            self,
            original_model_path: str,
            engine_path: str,
            onnx_path: Optional[str],
            model_config: Dict[str, Any],
            builder_config: Dict[str, Any],
    ) -> None:
        if onnx_path is None:
            raise ValueError("Потрібно передати шлях onnx_path")

        fp16_mode: bool = builder_config.get("fp16_mode", True)
        int8_mode: bool = builder_config.get("int8_mode", False)
        max_batch_size: int = builder_config.get("max_batch_size", 1)
        workspace_size_gb: int = builder_config.get("workspace_size_gb", 2)
        opset: int = builder_config.get("opset", 18)

        dynamic_h: tuple[int, int, int] = model_config.get(
            "dynamic_h", (32, 256, 1280)
        )
        dynamic_w: tuple[int, int, int] = model_config.get(
            "dynamic_w", (32, 256, 1280)
        )
        if any(len(t) != 3 for t in (dynamic_h, dynamic_w)):
            raise ValueError(
                "dynamic_h та dynamic_w мають бути кортежами з трьох елементів (min,opt,max)"
            )

        # ------------------------------------------------------------------
        # 1. Load TorchScript
        # ------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.jit.load(original_model_path, map_location=device).eval()
        except Exception as jit_err:
            warnings.warn(f"Не вдалося завантажити TorchScript модель '{original_model_path}': {jit_err}")
            try:
                obj = torch.load(original_model_path, map_location=device, weights_only=False)
                if isinstance(obj, torch.nn.Module):
                    model = obj.to(device)
                else:
                    raise RuntimeError(f"Непідтримуваний тип об’єкта з torch.load: {type(obj)}")
                model.eval()
            except Exception as load_err:
                raise RuntimeError(
                    f"Не вдалося завантажити модель '{original_model_path}' ані як TorchScript, "
                    f"ані через torch.load: {load_err}"
                ) from load_err

        if fp16_mode:
            if device.type == 'cuda':
                print("Конвертація завантаженої моделі в FP16...")
                model = model.half()
            else:
                warnings.warn("FP16 mode requested but running on CPU.")

        # ------------------------------------------------------------------
        # 2. Export to ONNX
        # ------------------------------------------------------------------
        example_input = _make_example_input(
            max_batch_size, dynamic_h, dynamic_w, device, fp16_mode
        )
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        print(f"[CRAFT] Exporting to ONNX → {onnx_path} (opset {opset})")
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["regions", "affinity"],
            dynamic_axes={  # enable dynamic shapes
                "images": {2: "height", 3: "width"},
                "regions": {2: "height", 3: "width"},
                "affinity": {2: "height", 3: "width"},
            },
        )
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
        print("[CRAFT] ONNX model is valid ✅")

        # ------------------------------------------------------------------
        # 3. Build TensorRT engine
        # ------------------------------------------------------------------
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1024 ** 3)
        )

        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)

        print(f"[CRAFT] Parsing ONNX {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                msgs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX parse failed:\n{msgs}")

        # ---- optimisation profile ----
        profile = builder.create_optimization_profile()
        image_name = "images"
        min_h, opt_h, max_h = dynamic_h
        min_w, opt_w, max_w = dynamic_w
        profile.set_shape(
            image_name,
            min=(1, 3, min_h, min_w),
            opt=(max_batch_size, 3, opt_h, opt_w),
            max=(max_batch_size, 3, max_h, max_w),
        )
        config.add_optimization_profile(profile)

        print("[CRAFT] Building serialized engine … (this may take a while)")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")

        # ------------------------------------------------------------------
        # 4. Save engine
        # ------------------------------------------------------------------
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        tmp_engine = f"{engine_path}.tmp"
        with open(tmp_engine, "wb") as f:
            f.write(engine_bytes)
        os.replace(tmp_engine, engine_path)
        print(f"[CRAFT] Engine saved → {engine_path}")

        # memory cleanup
        del engine_bytes, parser, network, config, builder, model, example_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    converter = CraftTrtConverter()
    converter.convert(
        original_model_path="./data/models/craft/detec_pt.pt",
        engine_path="./data/models/craft/craft_mlt_25k_2020-02-16.trt",
        onnx_path="./data/models/craft/craft_mlt_25k_2020-02-16.onnx",
        model_config={"dynamic_h": (32, 256, 1280), "dynamic_w": (32, 256, 1280)},
        builder_config={"fp16_mode": True, "workspace_size_gb": 4},
    )