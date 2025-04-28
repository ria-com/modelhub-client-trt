#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RefineNet TensorRT converter

Конвертує модель RefineNet (збережену як TorchScript .pt або .pth)
в ONNX, а потім в TensorRT engine (.trt), враховуючи специфічні
формати вхідних тензорів, які очікує затрейсена модель.

Приклад використання:
--------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python3.12 -m modelhub_client_trt.trt_converters.craft_refinenet

# Або імпортуйте та використовуйте клас:
# from refinenet_convertor import RefineNetTrtConverter
#
# converter = RefineNetTrtConverter()
# converter.convert(
#     original_model_path="path/to/your/refinenet.pt", # Шлях до .pt або .pth
#     engine_path="path/to/save/refinenet.trt",
#     onnx_path="path/to/save/refinenet.onnx",
#     # Важливо: dynamic_h/w тут - це розміри *оригінального* зображення для CRAFT
#     model_config={"dynamic_h": (32, 256, 1280), "dynamic_w": (32, 256, 1280)},
#     builder_config={"fp16_mode": True, "workspace_size_gb": 2, "max_batch_size": 1},
# )
--------------------------------------------------------------------
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Any, Optional, Tuple

import torch
import onnx
import tensorrt as trt
from modelhub_client_trt.trt_converters.base import BaseTrtConverter, TRT_LOGGER

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def _make_example_inputs_refinenet(
        bsz: int,
        dynamic_h: tuple[int, int, int], # Розміри оригінального зображення
        dynamic_w: tuple[int, int, int], # Розміри оригінального зображення
        device: torch.device,
        fp16: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Створює приклади вхідних тензорів для RefineNet, використовуючи
    оптимальні розміри (opt_h, opt_w) з динамічних діапазонів.

    Важливо:
    - input1 (y з CRAFT) генерується у форматі BHWC (Batch, H/2, W/2, Channels=2),
      оскільки затрейсена модель RefineNet.pt очікує саме такий формат
      і містить операцію permute для перетворення його в BCHW всередині.
    - input2 (feature з CRAFT) генерується у стандартному форматі BCHW
      (Batch, Channels=32, H/2, W/2).
    """
    _, opt_h, _ = dynamic_h
    _, opt_w, _ = dynamic_w

    # Розміри для RefineNet (половина від оригінальних)
    opt_h_half = opt_h // 2
    opt_w_half = opt_w // 2

    # Переконаємось, що розміри не нульові
    opt_h_half = max(1, opt_h_half)
    opt_w_half = max(1, opt_w_half)


    dtype = torch.float16 if fp16 and device.type == "cuda" else torch.float32

    # Вхід 1: формат BHWC (Batch, Height/2, Width/2, Channels=2)
    input1_shape = (bsz, opt_h_half, opt_w_half, 2)
    input1 = torch.randn(input1_shape, device=device, dtype=dtype)
    print(f"[Input Gen] Створено example_input1 (y) з формою {input1_shape} (BHWC)")

    # Вхід 2: формат BCHW (Batch, Channels=32, Height/2, Width/2)
    input2_shape = (bsz, 32, opt_h_half, opt_w_half)
    input2 = torch.randn(input2_shape, device=device, dtype=dtype)
    print(f"[Input Gen] Створено example_input2 (feature) з формою {input2_shape} (BCHW)")


    return input1, input2


# ----------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------


class RefineNetTrtConverter(BaseTrtConverter):
    """Конвертер RefineNet TorchScript/PyTorch -> ONNX -> TensorRT10."""

    def convert(  # noqa: C901
            self,
            original_model_path: str,
            engine_path: str,
            onnx_path: Optional[str],
            model_config: Dict[str, Any],
            builder_config: Dict[str, Any],
    ) -> None:
        """
        Виконує повний процес конвертації.

        Args:
            original_model_path: Шлях до вхідної моделі (.pt або .pth).
            engine_path: Шлях для збереження фінального TensorRT двигуна (.trt).
            onnx_path: Шлях для збереження проміжної ONNX моделі (.onnx).
            model_config: Словник конфігурації моделі (напр., dynamic_h/w).
            builder_config: Словник конфігурації TensorRT builder'а (напр., fp16, batch_size).
        """
        if onnx_path is None:
            raise ValueError("Потрібно передати шлях onnx_path для збереження ONNX моделі.")

        # --- Параметри конвертації ---
        fp16_mode: bool = builder_config.get("fp16_mode", True)
        int8_mode: bool = builder_config.get("int8_mode", False)
        max_batch_size: int = builder_config.get("max_batch_size", 1)
        workspace_size_gb: int = builder_config.get("workspace_size_gb", 2)
        opset: int = builder_config.get("opset", 18) # Рекомендовано 17+ для кращої сумісності

        # Динамічні розміри *оригінального* зображення (для CRAFT)
        dynamic_h: tuple[int, int, int] = model_config.get(
            "dynamic_h", (32, 256, 1280)
        )
        dynamic_w: tuple[int, int, int] = model_config.get(
            "dynamic_w", (32, 256, 1280)
        )
        if any(len(t) != 3 for t in (dynamic_h, dynamic_w)):
            raise ValueError(
                "dynamic_h та dynamic_w мають бути кортежами з трьох елементів (min,opt,max)"
                " і представляти розміри *оригінального* зображення."
            )
        print(f"[Config] Dynamic H (original image): {dynamic_h}")
        print(f"[Config] Dynamic W (original image): {dynamic_w}")
        print(f"[Config] Max Batch Size: {max_batch_size}")
        print(f"[Config] FP16 Mode: {fp16_mode}")
        print(f"[Config] INT8 Mode: {int8_mode}")
        print(f"[Config] Workspace (GB): {workspace_size_gb}")
        print(f"[Config] ONNX Opset: {opset}")


        # ------------------------------------------------------------------
        # 1. Load TorchScript або .pth Model
        # ------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Load] Використовується пристрій: {device}")
        model: Optional[torch.nn.Module] = None
        try:
            # Спочатку спробуємо завантажити як TorchScript (.pt)
            model = torch.jit.load(original_model_path, map_location=device).eval()
            print(f"[Load] Успішно завантажено TorchScript модель з: {original_model_path}")
        except Exception as jit_err:
            warnings.warn(f"[Load] Не вдалося завантажити як TorchScript '{original_model_path}': {jit_err}. Спроба завантажити як .pth...")
            try:
                # Спроба завантажити як звичайну PyTorch модель (.pth)
                # Потрібно, щоб клас RefineNet та copyStateDict були доступні
                try:
                    from refinenet import RefineNet # Припустимо, клас RefineNet тут
                except ImportError:
                     raise ImportError("Не вдалося імпортувати клас 'RefineNet'. Переконайтесь, що файл refinenet.py доступний.")
                try:
                     from pth2jit import copyStateDict # Припустимо, функція тут
                except ImportError:
                     # Проста альтернатива, якщо copyStateDict недоступний
                     warnings.warn("Не вдалося імпортувати 'copyStateDict'. Використовується стандартний load_state_dict.")
                     def copyStateDict(state_dict): return state_dict # Заглушка

                model_instance = RefineNet() # Створюємо інстанс моделі
                state_dict = torch.load(original_model_path, map_location='cpu') # Завантажуємо ваги на CPU
                # Обробляємо можливий префікс 'module.' якщо модель була збережена з DataParallel
                if isinstance(state_dict, dict) and all(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                    print("[Load] Видалено префікс 'module.' з ключів state_dict.")

                model_instance.load_state_dict(copyStateDict(state_dict)) # Копіюємо/завантажуємо ваги
                model = model_instance.to(device).eval()
                print(f"[Load] Успішно завантажено .pth модель з: {original_model_path}")

            except Exception as load_err:
                raise RuntimeError(
                    f"[Load] Помилка! Не вдалося завантажити модель '{original_model_path}' ані як TorchScript (.pt), "
                    f"ані через torch.load (.pth): {load_err}"
                ) from load_err

        if model is None:
             raise RuntimeError(f"Модель не була завантажена з {original_model_path}")

        # Конвертація в FP16 (якщо потрібно і можливо)
        if fp16_mode:
            if device.type == 'cuda':
                print("[Load] Конвертація завантаженої RefineNet моделі в FP16...")
                model = model.half()
            else:
                warnings.warn("[Load] FP16 mode requested для RefineNet, але немає CUDA. Пропускається конвертація в FP16.")

        # ------------------------------------------------------------------
        # 2. Export to ONNX
        # ------------------------------------------------------------------
        example_input1, example_input2 = _make_example_inputs_refinenet(
            max_batch_size, dynamic_h, dynamic_w, device, fp16_mode
        )
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        # Визначте імена вхідних та вихідних вузлів.
        # Ці імена мають відповідати тим, що використовуються у вашій моделі або бути узгодженими.
        input_names = ["craft_output_y", "craft_output_features"]
        # Дуже важливо перевірити реальне ім'я виходу RefineNet!
        # Припускаємо 'refined_affinity', але може бути іншим.
        output_names = ["refined_affinity"]

        print(f"[ONNX Export] Початок експорту в ONNX: {onnx_path} (opset {opset})")
        print(f"[ONNX Export] Імена входів: {input_names}")
        print(f"[ONNX Export] Імена виходів: {output_names}")

        try:
            torch.onnx.export(
                model,
                (example_input1, example_input2), # Входи як кортеж
                onnx_path,
                export_params=True,        # Зберігати ваги в ONNX файлі
                opset_version=opset,       # Версія ONNX opset
                do_constant_folding=True,  # Оптимізація (згортання констант)
                input_names=input_names,   # Назви вхідних вузлів
                output_names=output_names, # Назви вихідних вузлів
                dynamic_axes={             # Визначення динамічних осей
                    # Вхід 1 (очікується BHWC моделлю): осі H та W - це 1 та 2
                    input_names[0]: {1: "height_half", 2: "width_half"},
                    # Вхід 2 (BCHW): осі H та W - це 2 та 3
                    input_names[1]: {2: "height_half", 3: "width_half"},
                    # Вихід (припускаємо BCHW): осі H та W - це 2 та 3
                    output_names[0]: {2: "height_half", 3: "width_half"},
                },
                verbose=False # Встановіть True для детального логування ONNX експорту
            )
            print(f"[ONNX Export] Експорт в ONNX завершено успішно.")
        except torch.onnx.errors.UnsupportedOperatorError as op_err:
             print(f"\n[ONNX Export] Помилка! Модель містить оператор, не підтримуваний ONNX opset {opset}: {op_err}")
             print("Спробуйте змінити версію opset (напр., --opset 17) або змінити модель.")
             raise
        except Exception as export_err:
             print(f"\n[ONNX Export] Помилка під час експорту RefineNet в ONNX: {export_err}")
             # Виведемо форми тензорів-прикладів для діагностики
             print(f"    Форма example_input1: {example_input1.shape}, dtype: {example_input1.dtype}")
             print(f"    Форма example_input2: {example_input2.shape}, dtype: {example_input2.dtype}")
             raise RuntimeError(f"Помилка експорту RefineNet в ONNX: {export_err}") from export_err

        # Перевірка та збереження ONNX моделі
        print("[ONNX Save] Завантаження та перевірка створеної ONNX моделі...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX Save] ONNX модель валідна ✅")
        # Виправлення можливих проблем з великими моделями ONNX (>2GB)
        try:
            onnx.save_model(
                onnx_model,
                onnx_path,
                save_as_external_data=(onnx_model.ByteSize() > 2 * 1024**3) # Зберігати зовнішні дані якщо > 2GB
            )
            print(f"[ONNX Save] ONNX модель збережена в: {onnx_path}")
        except ValueError as save_err:
            print(f"[ONNX Save] Warning під час збереження ONNX: {save_err}. Модель може бути занадто великою.")
            # Спробувати зберегти без перевірки розміру якщо попереднє не вдалося
            onnx.save(onnx_model, onnx_path)
            print(f"[ONNX Save] ONNX модель збережена (альтернативний метод): {onnx_path}")


        # ------------------------------------------------------------------
        # 3. Build TensorRT engine
        # ------------------------------------------------------------------
        print("[TRT Build] Початок побудови TensorRT двигуна...")
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        config = builder.create_builder_config()

        # Налаштування пам'яті
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1024 ** 3)
        )
        print(f"[TRT Build] Встановлено ліміт Workspace: {workspace_size_gb} GB")

        # Налаштування точності (FP16/INT8)
        if fp16_mode:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[TRT Build] Увімкнено FP16 mode.")
            else:
                warnings.warn("[TRT Build] FP16 mode requested, але платформа його не підтримує швидко.")
        if int8_mode:
            if builder.platform_has_fast_int8:
                 config.set_flag(trt.BuilderFlag.INT8)
                 print("[TRT Build] Увімкнено INT8 mode.")
                 # Увага: INT8 зазвичай потребує калібрування для RefineNet
                 # Тут потрібно додати логіку для INT8 Calibrator, якщо він потрібен
                 warnings.warn("[TRT Build] Увімкнено INT8 mode, але калібратор не надано. Точність може постраждати.")
            else:
                 warnings.warn("[TRT Build] INT8 mode requested, але платформа його не підтримує швидко.")

        # Парсинг ONNX моделі
        print(f"[TRT Build] Парсинг ONNX моделі: {onnx_path}")
        with open(onnx_path, "rb") as f:
            onnx_content = f.read()
            if not parser.parse(onnx_content):
                print("\n[TRT Build] ПОМИЛКА парсингу ONNX!")
                for i in range(parser.num_errors):
                    print(f"    Error {i}: {parser.get_error(i)}")
                raise RuntimeError(f"Не вдалося розпарсити ONNX модель: {onnx_path}")
            else:
                 print("[TRT Build] ONNX модель успішно розпарсена.")


        # Створення Оптимізаційного Профілю для динамічних розмірів
        print("[TRT Build] Створення Оптимізаційного Профілю...")
        profile = builder.create_optimization_profile()

        # Розміри H/2, W/2 для профілю
        min_h, opt_h, max_h = dynamic_h
        min_w, opt_w, max_w = dynamic_w
        # Ділимо навпіл і гарантуємо, що мінімум >= 1
        min_h_half = max(1, min_h // 2)
        opt_h_half = max(1, opt_h // 2)
        max_h_half = max(1, max_h // 2)
        min_w_half = max(1, min_w // 2)
        opt_w_half = max(1, opt_w // 2)
        max_w_half = max(1, max_w // 2)

        print(f"  Профіль для '{input_names[0]}' (BHWC):")
        print(f"    Min Shape: (1, {min_h_half}, {min_w_half}, 2)")
        print(f"    Opt Shape: ({max_batch_size}, {opt_h_half}, {opt_w_half}, 2)")
        print(f"    Max Shape: ({max_batch_size}, {max_h_half}, {max_w_half}, 2)")
        profile.set_shape(
            input_names[0], # Ім'я першого входу
            min=(1, min_h_half, min_w_half, 2),          # Формат BHWC
            opt=(max_batch_size, opt_h_half, opt_w_half, 2), # Формат BHWC
            max=(max_batch_size, max_h_half, max_w_half, 2)  # Формат BHWC
        )

        print(f"  Профіль для '{input_names[1]}' (BCHW):")
        print(f"    Min Shape: (1, 32, {min_h_half}, {min_w_half})")
        print(f"    Opt Shape: ({max_batch_size}, 32, {opt_h_half}, {opt_w_half})")
        print(f"    Max Shape: ({max_batch_size}, 32, {max_h_half}, {max_w_half})")
        profile.set_shape(
            input_names[1], # Ім'я другого входу
            min=(1, 32, min_h_half, min_w_half),         # Формат BCHW
            opt=(max_batch_size, 32, opt_h_half, opt_w_half),# Формат BCHW
            max=(max_batch_size, 32, max_h_half, max_w_half) # Формат BCHW
        )
        config.add_optimization_profile(profile)

        # Побудова серіалізованого двигуна
        print("[TRT Build] Побудова серіалізованої мережі (може зайняти час)...")
        engine_bytes = builder.build_serialized_network(network, config)

        if engine_bytes is None:
            raise RuntimeError("[TRT Build] ПОМИЛКА! build_serialized_network повернув None.")
        else:
             print(f"[TRT Build] Побудова двигуна TensorRT завершена успішно.")


        # ------------------------------------------------------------------
        # 4. Save TensorRT engine
        # ------------------------------------------------------------------
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        tmp_engine_path = f"{engine_path}.tmp"
        try:
            with open(tmp_engine_path, "wb") as f:
                f.write(engine_bytes)
            os.replace(tmp_engine_path, engine_path) # Атомарне перейменування
            print(f"[TRT Save] Двигун TensorRT збережено: {engine_path}")
        except Exception as save_e:
             if os.path.exists(tmp_engine_path):
                 os.remove(tmp_engine_path)
             raise RuntimeError(f"Не вдалося зберегти двигун TensorRT: {save_e}") from save_e
        finally:
             # Очищення пам'яті
             del engine_bytes, parser, network, config, builder, model, example_input1, example_input2, onnx_model
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
                 print("[Cleanup] CUDA cache очищено.")


# ----------------------------------------------------------------------
# Main execution block (for standalone run)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print(" Запуск Конвертера RefineNet -> ONNX -> TensorRT ")
    print("="*60)

    # --- Налаштування шляхів ---
    # !!! ВАЖЛИВО: Вкажіть правильні шляхи до ваших файлів !!!
    # Шлях до вашої моделі RefineNet (формат .pt або .pth)
    # Наприклад, отриманий за допомогою pth2jit.py
    # default_refinenet_model_path = "../model_repository/refine_pt/1/refine_pt.pt" # Приклад шляху .pt
    default_refinenet_model_path = "./CRAFT-pytorch/model_repository/refine_pt/1/refine_pt.pt" # Приклад шляху .pth
    # Шляхи для збереження вихідних файлів
    default_onnx_save_path = "./data/models/craft/refinenet_fixed.onnx"
    default_trt_save_path = "./data/models/craft/refinenet_fixed.trt"

    # Ви можете перевизначити шляхи через аргументи командного рядка або змінити їх тут
    refinenet_model_path = os.environ.get("REFINENET_MODEL_PATH", default_refinenet_model_path)
    onnx_save_path = os.environ.get("ONNX_SAVE_PATH", default_onnx_save_path)
    trt_save_path = os.environ.get("TRT_SAVE_PATH", default_trt_save_path)

    print(f"Вхідна модель RefineNet: {refinenet_model_path}")
    print(f"Шлях для збереження ONNX: {onnx_save_path}")
    print(f"Шлях для збереження TRT : {trt_save_path}")

    # Перевірка існування файлу вхідної моделі
    if not os.path.exists(refinenet_model_path):
         print(f"\n!!! ПОМИЛКА: Вхідний файл моделі НЕ ЗНАЙДЕНО: {refinenet_model_path} !!!")
         print("Будь ласка, вкажіть правильний шлях до моделі RefineNet (.pt або .pth).")
         exit(1) # Вихід, оскільки конвертація неможлива

    # --- Налаштування конфігурації ---
    # Розміри оригінального зображення (ті самі, що й для CRAFT)
    # Виберіть діапазон, що відповідає вашим очікуваним вхідним даним
    model_cfg = {
        "dynamic_h": (32, 256, 1280), # (min_h, opt_h, max_h) оригінального зображення
        "dynamic_w": (32, 256, 1280)  # (min_w, opt_w, max_w) оригінального зображення
    }
    # Конфігурація побудови TensorRT
    builder_cfg = {
        "fp16_mode": True,         # Використовувати FP16 (рекомендовано)
        "int8_mode": False,        # Використовувати INT8 (потребує калібрування)
        "workspace_size_gb": 4,    # Більше пам'яті може пришвидшити побудову
        "max_batch_size": 1,       # Максимальний розмір батчу
        "opset": 18                # Версія ONNX opset
    }

    # --- Запуск конвертації ---
    converter = RefineNetTrtConverter()
    try:
        converter.convert(
            original_model_path=refinenet_model_path,
            engine_path=trt_save_path,
            onnx_path=onnx_save_path,
            model_config=model_cfg,
            builder_config=builder_cfg,
        )
        print("\n" + "="*60)
        print(" Конвертація RefineNet завершена УСПІШНО! ")
        print(f"  -> ONNX модель збережена в: {onnx_save_path}")
        print(f"  -> TRT двигун збережено в: {trt_save_path}")
        print("="*60)
    except FileNotFoundError as e:
        print(f"\n!!! ПОМИЛКА FileNotFoundError: {e} !!!")
        print("Перевірте правильність шляхів до файлів та існування директорій.")
    except ImportError as e:
         print(f"\n!!! ПОМИЛКА ImportError: {e} !!!")
         print("Переконайтесь, що необхідні файли (напр., refinenet.py, pth2jit.py) доступні у вашому PYTHONPATH.")
    except RuntimeError as e:
        print(f"\n!!! ПОМИЛКА RuntimeError під час конвертації RefineNet: {e} !!!")
        # Можна додати виведення повного стеку помилок для детальної діагностики
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n!!! НЕВІДОМА ПОМИЛКА під час конвертації RefineNet: {e} !!!")
        import traceback
        traceback.print_exc()
