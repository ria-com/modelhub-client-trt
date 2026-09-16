"""
CRNN (CTC, "STN-OCR" architecture) TensorRT converter.

Призначений для CRNN sequence-OCR моделей, у конфігу яких
`"tensorrt": {"type": "crnn"}` або `architecture` містить "STN-OCR"
(див. `auto_detect_converter_type` у `modelhub_client_trt.py`).

Архітектура: EfficientNet-B2 backbone -> Linear(5632, 512) -> 2x bidirectional
LSTM (атрибут називається `gru`) -> Linear -> CTC logits. Softmax/CTC-decode у
граф не запікаються — декодування виконується на стороні клієнта.

Артефакт моделі — PyTorch Lightning checkpoint (dict з ключами `state_dict` і
`hyper_parameters`), а не TorchScript і не pickled `nn.Module`. Тому `convert()`
відтворює архітектуру (`_CrnnOcrNet`) і завантажує `state_dict` напряму, без
залежності від `pytorch_lightning` чи коду тренування. Імена тензорів у
`state_dict` мають збігатися з підмодулями `_CrnnOcrNet`.

Ручний тест (потрібен GPU):
CUDA_VISIBLE_DEVICES=0 python3 -m modelhub_client_trt.trt_converters.crnn
"""

from __future__ import annotations

import inspect
import os
import pickle
import types
import warnings
from typing import Any, Dict, Optional

import onnx
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2 as _efficientnet_b2

from .base import BaseTrtConverter, build_engine_from_onnx


class _StubObject:
    """Заглушка для класів, яких немає в поточному середовищі (див. `_load_checkpoint`)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_StubObject,), {"__module__": module})


_tolerant_pickle = types.ModuleType("_tolerant_pickle")
_tolerant_pickle.Unpickler = _TolerantUnpickler
_tolerant_pickle.load = lambda file, **kwargs: _TolerantUnpickler(file, **kwargs).load()


def _load_checkpoint(path: str) -> Any:
    """
    Завантажує Lightning checkpoint. Такі чекпоінти часто містять у
    `hyper_parameters` запікльовані об'єкти з коду тренування (напр. власні
    label-converter класи), яких немає там, де виконується конвертація.
    Конвертеру вони не потрібні (лише `state_dict` і прості гіперпараметри),
    тому при відсутньому модулі/класі вони підміняються заглушками.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, AttributeError) as e:
        warnings.warn(
            f"(CRNN) Checkpoint посилається на класи, недоступні в цьому середовищі ({e}); "
            f"вони будуть замінені заглушками."
        )
        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_tolerant_pickle)


class _BlockRNN(nn.Module):
    """
    Bidirectional-LSTM блок. Атрибут `gru` (фактично `nn.LSTM`) збережено
    заради сумісності імен параметрів у `state_dict` чекпоінта.
    """

    def __init__(self, in_size: int, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.LSTM(in_size, hidden_size, bidirectional=bidirectional, batch_first=True)

    def forward(self, batch: torch.Tensor, add_output: bool = False) -> torch.Tensor:
        outputs, _hidden = self.gru(batch)
        if add_output:
            out_size = outputs.size(2) // 2
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs


class _CrnnOcrNet(nn.Module):
    """
    Inference-only реконструкція CRNN (EfficientNet-B2 -> Linear -> 2x
    `_BlockRNN` -> Linear -> CTC logits) без залежності від `pytorch_lightning`.

    Лінійний шар `Linear(5632, 512)` фіксує розмір входу: для (N, 3, 128, 384)
    backbone дає `(N, 12, 5632)`. Вихід: `[seq_len, batch, num_classes]`, де
    `num_classes = len(letters) + 1` (CTC blank).
    """

    def __init__(self, letters_max: int, hidden_size: int = 32, bidirectional: bool = True):
        super().__init__()
        backbone = _efficientnet_b2(weights=None)
        # Drop AdaptiveAvgPool2d + classifier head; keep the conv feature
        # extractor ("features") only.
        self.backbone_app = nn.Sequential(*list(backbone.children())[:-2])
        self.linear1 = nn.Linear(5632, 512)
        self.gru1 = _BlockRNN(512, hidden_size, bidirectional=bidirectional)
        self.gru2 = _BlockRNN(hidden_size, hidden_size, bidirectional=bidirectional)
        self.linear2 = nn.Linear(hidden_size * 2, letters_max)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.size(0)

        batch = self.backbone_app(batch)

        # make sequences of image features
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.reshape(batch_size, n_channels, -1)

        batch = self.linear1(batch)

        batch = self.gru1(batch, add_output=True)
        batch = self.gru2(batch)

        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)  # -> [seq_len, batch, num_classes]; raw CTC logits
        return batch


class CrnnTrtConverter(BaseTrtConverter):
    """Конвертер TensorRT для CRNN/CTC OCR-моделей (architecture == 'STN-OCR')."""

    def convert(self,
                original_model_path: str,
                engine_path: str,
                onnx_path: Optional[str],
                model_config: Dict[str, Any],
                builder_config: Dict[str, Any]) -> None:
        if not onnx_path:
            raise ValueError("Для CrnnTrtConverter потрібен шлях до ONNX файлу (onnx_path).")

        fp16_mode = builder_config.get('fp16_mode', True)
        max_batch_size = builder_config.get('max_batch_size', 1)
        memory_limit = builder_config.get('memory_limit')
        opset = builder_config.get('opset', 19)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. Завантажити checkpoint (PyTorch Lightning dict, НЕ TorchScript) ---
        print(f"(CRNN) Завантаження checkpoint з: {original_model_path}")
        checkpoint = _load_checkpoint(original_model_path)
        if not (isinstance(checkpoint, dict) and "state_dict" in checkpoint):
            raise RuntimeError(
                f"CrnnTrtConverter очікує PyTorch Lightning checkpoint "
                f"(dict з ключем 'state_dict'), отримано: {type(checkpoint)}"
            )
        state_dict = checkpoint["state_dict"]
        hyper_parameters = checkpoint.get("hyper_parameters", {}) or {}

        letters = model_config.get("letters") or hyper_parameters.get("letters") or []
        letters_max = hyper_parameters.get("letters_max") or (len(letters) + 1 if letters else None)
        if not letters_max:
            raise ValueError(
                "Не вдалося визначити letters_max (ні з checkpoint['hyper_parameters'], "
                "ні з model_config['letters'])."
            )
        hidden_size = hyper_parameters.get("hidden_size", 32)
        bidirectional = hyper_parameters.get("bidirectional", True)

        # --- 2. Реконструювати архітектуру та завантажити ваги ---
        model = _CrnnOcrNet(letters_max=letters_max, hidden_size=hidden_size, bidirectional=bidirectional)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"CRNN checkpoint state_dict не збігається з реконструйованою архітектурою "
                f"_CrnnOcrNet. missing={missing}, unexpected={unexpected}"
            )
        model = model.to(device).eval()

        if fp16_mode:
            if device.type == 'cuda':
                print("Конвертація завантаженої моделі в FP16...")
                model = model.half()
            else:
                warnings.warn("FP16 mode requested but running on CPU.")

        # --- 3. Підготувати вхідні дані (фіксована статична форма) ---
        image_h = model_config.get("image_size_h", 128)
        image_w = model_config.get("image_size_w", 384)
        color_channels = model_config.get("color_channels", 3)

        dtype = torch.half if fp16_mode and device.type == 'cuda' else torch.float
        example_input = torch.randn(max_batch_size, color_channels, image_h, image_w, device=device, dtype=dtype)
        print(f"(CRNN) Приклад вхідних даних для ONNX ({example_input.dtype}) з формою: "
              f"{tuple(example_input.shape)} на {device}")

        # --- 4. Експорт в ONNX ---
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        input_names = ["images"]
        output_names = ["logits"]
        export_kwargs = dict(
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,  # статична форма — узгоджено з build_engine_from_onnx
            verbose=False,
        )
        print(f"(CRNN) Експорт в ONNX (opset {opset}) у файл: {onnx_path}...")
        # cuDNN вимикається лише на час трейсингу/експорту: у деяких збірках torch/cuDNN
        # відсутня libcudnn_engines_runtime_compiled.so і Conv2d.forward на CUDA падає з
        # "CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED". На сам TensorRT-двигун це не впливає.
        with torch.backends.cudnn.flags(enabled=False):
            try:
                with torch.no_grad():
                    torch.onnx.export(model, example_input, onnx_path, **export_kwargs)
            except Exception as e:
                # Dynamo-based exporter (замовчування в нових torch) у деяких комбінаціях
                # версій torch/onnx_ir падає ще до експорту (напр. "module 'onnx_ir' has no
                # attribute 'schemas'"). Відкат на legacy exporter (dynamo=False).
                sig = inspect.signature(torch.onnx.export)
                if "dynamo" not in sig.parameters:
                    raise RuntimeError(f"Помилка під час експорту CRNN моделі в ONNX: {e}") from e
                warnings.warn(
                    f"torch.onnx.export з dynamo=True (замовчування) провалився ({e}). "
                    f"Повтор експорту через legacy exporter (dynamo=False)..."
                )
                if os.path.exists(onnx_path):
                    try:
                        os.remove(onnx_path)
                    except OSError:
                        pass
                try:
                    with torch.no_grad():
                        torch.onnx.export(model, example_input, onnx_path, dynamo=False, **export_kwargs)
                except Exception as e2:
                    raise RuntimeError(
                        f"Помилка під час експорту CRNN моделі в ONNX навіть з legacy exporter: {e2}"
                    ) from e2

        print(f"(CRNN) Перевірка створеної ONNX моделі: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("(CRNN) ONNX модель пройшла перевірку.")

        # --- 5. Побудова TensorRT двигуна (спільний білдер з base.py) ---
        build_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=engine_path,
            fp16_mode=fp16_mode,
            max_batch_size=max_batch_size,
            memory_limit=memory_limit,
        )
        print(f"(CRNN) TensorRT двигун збережено: {engine_path}")

        del model, example_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    converter = CrnnTrtConverter()
    converter.convert(
        original_model_path="./data/models/crnn_ocr/model.ckpt",
        engine_path="./data/models/crnn_ocr/model.engine",
        onnx_path="./data/models/crnn_ocr/model.onnx",
        model_config={
            "letters": [".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "image_size_h": 128,
            "image_size_w": 384,
        },
        builder_config={
            "fp16_mode": True,
            "max_batch_size": 1,
            "memory_limit": 2 * (1024 ** 3),
            "opset": 19,
        },
    )
