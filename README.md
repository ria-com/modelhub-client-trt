# RIA ModelHub Client with TensorRT support
This class extends the capabilities of the https://github.com/ria-com/modelhub-client module  

## Installation
```bash
python3.12 -m pip install git+https://github.com/ria-com/modelhub-client-trt.git
```
or
```bash
git clone https://github.com/ria-com/modelhub-client-trt.git
cd ./modelhub-client-trt
python3.12 setup.py install
```

## Quick Start
An example of usage can be seen in the examples folder.


## Engine cache and build modes (1.4.0)

An engine depends on the TensorRT version and the GPU model it was built on, not on the driver.
Each engine gets a manifest `<engine>.json`; `download_model_by_name_trt` returns it as
`model_info["trt_manifest"]` and reuses the engine only if the manifest matches the current GPU
and TensorRT (otherwise the engine is rebuilt, or refused in `never` mode).

| `MODELHUB_TRT_BUILD` | behaviour |
|---|---|
| `ifmissing` (default) | reuse a matching engine, build otherwise |
| `never` | reuse a matching engine, raise `TrtEngineMissing` otherwise (serving pods) |
| `force` | always rebuild |

Typical setup: a build Job per GPU model runs with `ifmissing`, serving pods run with `never`,
and `gpu_memory_estimate(manifest)` (weights + device memory) sizes the GPU memory limit.
