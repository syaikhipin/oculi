from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    license: str
    note: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        name="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license="MIT (OpenCV Zoo, Shiqi Yu)",
        note="face detection; commercial use permitted",
    ),
    "sface": ModelSpec(
        name="sface",
        filename="face_recognition_sface_2021dec.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        license="Apache-2.0 (OpenCV Zoo)",
        note="128-D face embeddings; commercial use permitted",
    ),
    "yolox": ModelSpec(
        name="yolox",
        filename="object_detection_yolox_2022nov.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        license="Apache-2.0 (OpenCV Zoo)",
        note="YOLOX-s, COCO-80; used for person/vehicle classification",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "oculi/0.2"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_model(spec: ModelSpec, model_dir: Path, allow_download: bool = True) -> Path:
    path = model_dir / spec.name / spec.filename
    if path.exists() and sha256_file(path) == spec.sha256:
        return path
    if path.exists():
        if not allow_download:
            raise RuntimeError(f"Model checksum mismatch: {path}")
        path.unlink()
    if not allow_download:
        raise FileNotFoundError(f"Missing model {path}")
    print(f"downloading {spec.name} -> {path}", file=sys.stderr)
    _download(spec.url, path)
    actual = sha256_file(path)
    if actual != spec.sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {spec.name}: {actual} != {spec.sha256}")
    return path


def ensure_models(model_dir: Path, allow_download: bool = True) -> dict[str, Path]:
    return {name: ensure_model(spec, model_dir, allow_download) for name, spec in MODEL_SPECS.items()}


def model_info() -> list[dict[str, str]]:
    items = [
        {
            "name": spec.name,
            "filename": spec.filename,
            "sha256": spec.sha256,
            "license": spec.license,
            "url": spec.url,
            "note": spec.note,
        }
        for spec in MODEL_SPECS.values()
    ]
    items.append(
        {
            "name": "insightface-buffalo_l",
            "filename": "det_10g.onnx + w600k_r50.onnx (+ landmarks/genderage)",
            "sha256": "downloaded and unpacked by insightface",
            "license": "code MIT; pretrained models non-commercial/research (see InsightFace model zoo)",
            "url": "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
            "note": "512-D face embeddings",
        }
    )
    return items
