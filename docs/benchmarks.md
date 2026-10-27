# Oculi GPU benchmark

Generated: `2026-10-27T17:20:57Z` on **Tesla T4** (Modal), ONNX Runtime `1.22.0`.

Providers: `TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider`.

## Face detection + 512-D embedding (InsightFace)

| Input | CPU ms/img | CUDA ms/img | Speed-up | CUDA images/s |
|---|---:|---:|---:|---:|
| 640x480 | 130.1 | 36.7 | 3.5× | 27.2 |
| 1280x720 | 141.9 | 32.7 | 4.3× | 30.6 |
| 1920x1080 | 147.8 | 32.9 | 4.5× | 30.4 |

## Other stages

| Stage | Mean ms | FPS | Provider |
|---|---:|---:|---|
| Passive liveness (MiniFASNet, CPU session) | 5.7 | 174.4 | CPU |
| YOLOX objects (ONNX Runtime) | 32.5 | 30.8 | CUDAExecutionProvider, CPUExecutionProvider |
| ALPR (fast-alpr) | 26.2 | 38.1 | ONNX Runtime |
| Full `analyze()` on street image | 130.1 | 7.7 | mixed |
| Face search (detect+embed+FAISS) | 60.0 | 16.7 | CUDA |

## Notes

- Face detection/embedding and YOLOX use CUDA via `onnxruntime-gpu`; liveness keeps a CPU session and ALPR uses ONNX Runtime.
- YOLOX now runs through ONNX Runtime; on a CUDA host it selects `CUDAExecutionProvider`.
- `TensorrtExecutionProvider` is available in the image but not explicitly forced; the first CUDA run wins.
- ALPR and liveness timings are measured through their current providers on the same T4 host.
- Results are machine-specific; rerun with `modal run benchmarks/modal_gpu_bench.py`.

Raw JSON: `gpu_results.json`.
