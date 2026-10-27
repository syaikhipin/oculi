"""GPU benchmark for the Oculi CV stack on Modal.

Run:
    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... modal run benchmarks/modal_gpu_bench.py

Models are cached in the `oculi-models` Modal volume; results are written locally to
benchmarks/gpu_results.json and docs/benchmarks.md.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

import modal

APP_NAME = "oculi-gpu-bench"
CUDA_IMAGE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("build-essential", "libgl1", "libglib2.0-0", "libgomp1", "curl", "clang")
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "onnxruntime-gpu==1.22.0",
        "insightface==0.7.3",
        "cython",
        "fast-alpr==0.4.0",
        "fast-plate-ocr==1.1.0",
        "open-image-models==0.6.0",
        "onnx==1.17.0",
        "scikit-image",
        "requests",
        "tqdm",
    )
    .env({"PYTHONPATH": "/app/src", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"})
    .add_local_dir("src", remote_path="/app/src")
)

volume = modal.Volume.from_name("oculi-models", create_if_missing=True)
app = modal.App(APP_NAME)


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def _bench(fn, iterations: int, warmup: int = 2) -> dict:
    import numpy as np

    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "iterations": iterations,
        "mean_ms": round(statistics.mean(samples), 2),
        "p50_ms": round(statistics.median(samples), 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
        "fps": round(1000.0 / statistics.mean(samples), 2),
    }


@app.function(gpu="T4", image=image, volumes={"/models": volume}, timeout=1800)
def bench() -> dict:
    import cv2
    import numpy as np
    import onnxruntime as ort

    from oculi.config import Settings
    from oculi.cv.alpr import AlprEngine
    from oculi.cv.liveness import LivenessEngine
    from oculi.cv.objects import ObjectEngine
    from oculi.engine import create_face_engine
    from oculi.pipeline import FramePipeline
    from oculi.service import FaceService

    print("onnxruntime providers:", ort.get_available_providers())
    print("onnxruntime version:", ort.__version__)

    data = Path("/tmp/oculi-bench")
    data.mkdir(parents=True, exist_ok=True)
    images = {
        "face": data / "obama.jpg",
        "face2": data / "biden.jpg",
        "street": data / "bus.jpg",
        "plate": data / "plate.png",
    }
    _download("https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg", images["face"])
    _download("https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg", images["face2"])
    _download("https://ultralytics.com/images/bus.jpg", images["street"])
    _download("https://raw.githubusercontent.com/ankandrew/fast-alpr/master/assets/test_image.png", images["plate"])

    def settings(provider: str) -> Settings:
        return Settings(
            data_dir=data,
            model_dir=Path("/models"),
            db_path=data / "bench.db",
            face_provider=provider,
            face_engine="insightface",
            use_faiss=True,
        )

    # -------- face detect+embed, CPU vs CUDA --------
    cpu_engine = create_face_engine(settings("cpu"))
    gpu_engine = create_face_engine(settings("cuda"))
    base = cv2.imread(str(images["face"]))
    sizes = {"640x480": (640, 480), "1280x720": (1280, 720), "1920x1080": (1920, 1080)}
    face_results = {"providers": {"cpu": cpu_engine.info()["providers"], "cuda": gpu_engine.info()["providers"]}, "sizes": {}}
    for label, (width, height) in sizes.items():
        frame = cv2.resize(base, (width, height), interpolation=cv2.INTER_AREA)
        face_results["sizes"][label] = {
            "cpu": _bench(lambda f=frame: cpu_engine.detect(f), 3),
            "cuda": _bench(lambda f=frame: gpu_engine.detect(f), 15),
        }

    # -------- per-stage on GPU machine --------
    liveness = LivenessEngine(Path("/models/liveness"))
    objects = ObjectEngine(settings("cuda"))
    alpr = AlprEngine(Path("/models/alpr"), conf=0.4)
    street = cv2.imread(str(images["street"]))
    face_frame = cv2.imread(str(images["face"]))
    plate_frame = cv2.imread(str(images["plate"]))
    faces = gpu_engine.detect(face_frame)
    bbox = faces[0].bbox if faces else (10, 10, 100, 100)

    stages = {
        "liveness_cpu": _bench(lambda: liveness.check(face_frame, bbox), 10),
        "objects_yolox": _bench(lambda: objects.detect(street), 10),
        "objects_providers": objects.info()["providers"],
        "alpr": _bench(lambda: alpr.detect(plate_frame), 10),
    }

    # -------- full pipeline + matching --------
    service = FaceService(settings("cuda"))
    service.enroll("default", "obama", "Barack Obama", [images["face"].read_bytes()], require_quality=False)
    pipeline = FramePipeline(settings("cuda"), service)
    pipeline.analyze(street)  # warmup
    full = _bench(lambda: pipeline.analyze(street), 5)

    search_frame = cv2.imread(str(images["face2"]))
    match = _bench(lambda: service.search("default", cv2.imencode(".jpg", search_frame)[1].tobytes()), 5)

    # Model volume sizes (cached across runs).
    model_files = sorted(
        (str(p.relative_to("/models")), p.stat().st_size)
        for p in Path("/models").rglob("*")
        if p.is_file()
    )
    return {
        "gpu": os.environ.get("MODAL_GPU", "Tesla T4"),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "face": face_results,
        "stages": stages,
        "full_pipeline_1080p": full,
        "matching_search": match,
        "models": [{"path": path, "bytes": size} for path, size in model_files],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.local_entrypoint()
def main() -> None:
    result = bench.remote()
    root = Path(__file__).resolve().parents[1]
    (root / "benchmarks").mkdir(exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "benchmarks" / "gpu_results.json").write_text(json.dumps(result, indent=2) + "\n")

    rows = []
    for size, values in result["face"]["sizes"].items():
        cpu = values["cpu"]["mean_ms"]
        cuda = values["cuda"]["mean_ms"]
        rows.append(f"| {size} | {cpu:.1f} | {cuda:.1f} | {cpu / cuda:.1f}× | {values['cuda']['fps']:.1f} |")
    stages = result["stages"]
    md = f"""# Oculi GPU benchmark

Generated: `{result['timestamp_utc']}` on **{result['gpu']}** (Modal), ONNX Runtime `{result['onnxruntime']}`.

Providers: `{', '.join(result['providers'])}`.

## Face detection + 512-D embedding (InsightFace)

| Input | CPU ms/img | CUDA ms/img | Speed-up | CUDA images/s |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Other stages

| Stage | Mean ms | FPS | Provider |
|---|---:|---:|---|
| Passive liveness (MiniFASNet, CPU session) | {stages['liveness_cpu']['mean_ms']:.1f} | {stages['liveness_cpu']['fps']:.1f} | CPU |
| YOLOX objects (ONNX Runtime) | {stages['objects_yolox']['mean_ms']:.1f} | {stages['objects_yolox']['fps']:.1f} | {', '.join(stages['objects_providers'])} |
| ALPR (fast-alpr) | {stages['alpr']['mean_ms']:.1f} | {stages['alpr']['fps']:.1f} | ONNX Runtime |
| Full `analyze()` on street image | {result['full_pipeline_1080p']['mean_ms']:.1f} | {result['full_pipeline_1080p']['fps']:.1f} | mixed |
| Face search (detect+embed+FAISS) | {result['matching_search']['mean_ms']:.1f} | {result['matching_search']['fps']:.1f} | CUDA |

## Notes

- Face detection/embedding and YOLOX use CUDA via `onnxruntime-gpu`; liveness keeps a CPU session and ALPR uses ONNX Runtime.
- YOLOX now runs through ONNX Runtime; on a CUDA host it selects `CUDAExecutionProvider`.
- `TensorrtExecutionProvider` is available in the image but not explicitly forced; the first CUDA run wins.
- ALPR and liveness timings are measured through their current providers on the same T4 host.
- Results are machine-specific; rerun with `modal run benchmarks/modal_gpu_bench.py`.

Raw JSON: `gpu_results.json`.
"""
    (root / "docs" / "benchmarks.md").write_text(md)
    print(json.dumps(result, indent=2))
