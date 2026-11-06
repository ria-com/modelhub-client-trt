Changelog for `modelhub-client`
=======================

## Release 1.4.0
* Engine manifest `<engine>.json` next to every engine: GPU name, compute capability, full TensorRT version, engine size, device memory, build time, library version. An engine is reused only when its manifest matches the current GPU and TensorRT; an engine without a manifest (built by an older release) is checked by deserialization once and adopted.
* An engine that does not deserialize or was built for another TensorRT version is rebuilt instead of being handed to the caller.
* `MODELHUB_TRT_BUILD` = `ifmissing` (default, as before) | `never` (raise `TrtEngineMissing` instead of building — serving pods, engines come from a build Job) | `force`.
* Build lock with a heartbeat: a lock left by a killed process is taken over after 120 s; waiters wait while the owner is alive (`trt_build_wait_timeout` default 3600 s instead of 300 s).
* Cleanup after a build keeps manifests, other processes' locks and temporary files.
* `read_manifest`, `gpu_memory_estimate` (engine weights + device memory) for sizing GPU memory limits.

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
