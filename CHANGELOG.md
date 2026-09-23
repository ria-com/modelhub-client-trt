Changelog for `modelhub-client`
=======================

## Release 1.3.0
* `tensorrt.dynamic_hw` model config flag: classic YOLOv5 (torch.hub) checkpoints are exported with dynamic height/width and the engine gets an optimization profile up to `imgsz`, so it accepts the same rectangular letterbox as torch.hub AutoShape; the engine file name gets a `-dynhw` suffix.
* `build_engine_from_onnx(..., max_hw=(H, W))` for rank-4 inputs with dynamic batch/height/width.

## Release 1.2.0
* New `crnn` converter for CTC CRNN OCR models (PyTorch Lightning checkpoints).
* `tensorrt.nms` model config flag: bake NMS into exported YOLO engines; the engine file name gets a `-nms` suffix.
* YOLO converter: fallback paths for classic YOLOv5 (torch.hub) checkpoints and for TensorRT bindings incompatible with `ultralytics` export; yolov5 repo caching runs in a subprocess.
* Converter auto-detection by `architecture` (`yolo` -> `yolo`, `stn-ocr` -> `crnn`) when `tensorrt.type` is not set.
* Compatibility with newer TensorRT/PyTorch: optional `EXPLICIT_BATCH` / precision flags, `dynamo=False` ONNX export for CRAFT models, ONNX parsing with external weight data.
* Shared `build_engine_from_onnx` helper; unit tests.

## Release 0.0.1
* Pip module installation
