Changelog for `modelhub-client`
=======================

## Release 1.4.4
* `onnxslim` is a requirement again: ultralytics needs it for `export(format="engine")`, and 1.4.3 dropped it.
* The YOLO converter sets `YOLO_AUTOINSTALL=False` (unless set): ultralytics used to pip-install missing packages inside the running build, which in the cluster broke numpy and produced an empty ONNX ("Network must have at least one output").

## Release 1.4.3
* Install requirements are only `onnx` and `modelhub_client`: torch and tensorrt come from the base image. Requiring `torch` made pip replace the NGC pre-release torch (`2.6.0a0+…nv24.12`) with a PyPI wheel and break the image's torchvision (`operator torchvision::nms does not exist`). Unused `torch2trt`, `onnxruntime-gpu`, `onnxslim` dropped.
* `setup.py` split requirements on `"/n"` instead of newlines.

## Release 1.4.2
* An engine that TensorRT deserializes but reports as built on another GPU model ("plan file across different models of devices") is treated as unusable and rebuilt, instead of being adopted.
* Manifests get `device_checked`; manifests written by 1.4.0/1.4.1 are re-checked once (engines adopted by build Jobs before this fix included).

## Release 1.4.1
* `crnn` converter: layer sizes come from the checkpoint weights (`linear1` in/out, LSTM hidden size, bidirectional, number of classes) instead of the hard-coded `Linear(5632, 512)`; the input size from the model config (`image_size_h/w`, then `height/width` as OCR configs write it, then the checkpoint `height/width`, then the old 128x384). A wrong input size fails with a clear error before export. Checkpoints with `conv_nn` / `recurrent_layer{1,2}.rnn` module names (cadastral-number-detection, nomeroff-net) are mapped onto the same graph. Example: ria-vin-detection OCR — `Linear(2816, 512)`, 64x2048.
* Converter auto-detection: a `.ckpt` file with `letters` in the model config is a CRNN OCR checkpoint (`crnn`), not an image classifier.
* `image_classifier`: when the dynamo ONNX exporter (default in torch >= 2.9) fails inside onnx_ir/onnxscript/torch.export, the export is retried with the legacy exporter, as already done for TorchScript models.

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
