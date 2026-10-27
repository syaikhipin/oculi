from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .models import MODEL_SPECS, ensure_model

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # pragma: no cover
    pass
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")

SFACE_ENGINE_ID = "sface-128-v1"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
INSIGHTFACE_ENGINE_ID = "insightface-buffalo_l-v1"
INSIGHTFACE_MODEL = "w600k_r50.onnx"


@dataclass
class ExtractedFace:
    vector: np.ndarray
    bbox: tuple[int, int, int, int]
    det_score: float
    blur: float
    landmarks: list[tuple[float, float]]
    sex: str | None = None
    age: int | None = None
    attributes: dict = field(default_factory=dict)

    def to_dict(self, include_vector: bool = False) -> dict:
        data = {
            "bbox": list(self.bbox),
            "det_score": round(float(self.det_score), 6),
            "blur": round(float(self.blur), 3),
            "landmarks": [[round(float(x), 3), round(float(y), 3)] for x, y in self.landmarks],
            "sex": self.sex,
            "age": self.age,
            "attributes": self.attributes,
        }
        if include_vector:
            data["vector"] = self.vector.astype(float).tolist()
        return data


class FaceEngine:
    """Commercial-friendly default: YuNet (MIT) + SFace (Apache-2.0), 128-D.

    OpenCV's FaceRecognizerSF performs the 5-point similarity alignment internally.
    """

    def __init__(self, settings: Settings, allow_model_download: bool = True) -> None:
        self.settings = settings
        settings.ensure_dirs()
        det_path = ensure_model(MODEL_SPECS["yunet"], settings.model_dir, allow_download=allow_model_download)
        rec_path = ensure_model(MODEL_SPECS["sface"], settings.model_dir, allow_download=allow_model_download)
        self._detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), settings.det_score, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self._lock = threading.Lock()

    @property
    def engine_id(self) -> str:
        return SFACE_ENGINE_ID

    def info(self) -> dict:
        return {
            "engine": SFACE_ENGINE_ID,
            "detector": "face_detection_yunet_2023mar.onnx (MIT)",
            "recognizer": SFACE_MODEL,
            "embedding_dim": 128,
            "template_bytes": 128 * 4,
            "det_score": self.settings.det_score,
            "min_face_px": self.settings.min_face_px,
            "license": "YuNet MIT, SFace Apache-2.0 (commercial use permitted)",
        }

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("empty image payload")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not decode image (expected JPEG/PNG/BMP/WebP)")
        return image

    def detect(self, image: np.ndarray) -> list[ExtractedFace]:
        height, width = image.shape[:2]
        with self._lock:
            self._detector.setInputSize((width, height))
            _, rows = self._detector.detect(image)
            if rows is None:
                return []
            results: list[ExtractedFace] = []
            for row in rows:
                x, y, w, h = (float(v) for v in row[:4])
                if min(w, h) < self.settings.min_face_px:
                    continue
                x1 = max(0, min(width - 1, int(round(x))))
                y1 = max(0, min(height - 1, int(round(y))))
                x2 = max(0, min(width, int(round(x + w))))
                y2 = max(0, min(height, int(round(y + h))))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = image[y1:y2, x1:x2]
                blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                aligned = self._recognizer.alignCrop(image, row)
                feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(feature))
                if not np.isfinite(norm) or norm < 1e-12:
                    continue
                landmarks = [(float(px), float(py)) for px, py in np.asarray(row[4:14]).reshape(5, 2)]
                results.append(
                    ExtractedFace(
                        vector=feature / norm,
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        det_score=float(row[14]),
                        blur=blur,
                        landmarks=landmarks,
                    )
                )
        results.sort(key=lambda item: item.bbox[2] * item.bbox[3], reverse=True)
        return results

    def extract(self, image, select: str = "largest", require_quality: bool = False) -> list[ExtractedFace]:
        if isinstance(image, bytes):
            image = self.decode_image(image)
        faces = self.detect(image)
        if require_quality:
            faces = [face for face in faces if face.blur >= self.settings.blur_min]
        if select == "largest":
            return faces[:1]
        if select != "all":
            raise ValueError("select must be 'largest' or 'all'")
        return faces

    def extract_one(self, image, require_quality: bool = True) -> ExtractedFace:
        faces = self.extract(image, select="largest", require_quality=require_quality)
        if not faces:
            raise ValueError("no usable face detected")
        return faces[0]

    def extract_path(self, path: str | Path, select: str = "largest") -> list[ExtractedFace]:
        return self.extract(Path(path).read_bytes(), select=select)


class InsightFaceEngine:
    """Optional higher-accuracy engine. NOT for commercial use by default:
    InsightFace buffalo_l pretrained weights are research/non-commercial."""

    def __init__(self, settings: Settings, allow_model_download: bool = True) -> None:
        self.settings = settings
        settings.ensure_dirs()
        root = settings.model_dir / "insightface"
        model_path = root / "models" / "buffalo_l" / "det_10g.onnx"
        if not model_path.exists() and not allow_model_download:
            raise FileNotFoundError(f"InsightFace models not found at {model_path}")
        providers = self._providers(settings.face_provider)
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(name="buffalo_l", root=str(root), providers=providers)
        ctx_id = 0 if any("CUDA" in provider for provider in providers) else -1
        self._app.prepare(ctx_id=ctx_id, det_thresh=float(settings.det_score), det_size=(settings.face_det_size, settings.face_det_size))
        self._lock = threading.Lock()

    @staticmethod
    def _providers(preference: str) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if preference == "cuda":
            return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if preference == "cpu":
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    @property
    def engine_id(self) -> str:
        return INSIGHTFACE_ENGINE_ID

    def info(self) -> dict:
        return {
            "engine": INSIGHTFACE_ENGINE_ID,
            "detector": "det_10g.onnx (SCRFD)",
            "recognizer": INSIGHTFACE_MODEL,
            "embedding_dim": 512,
            "template_bytes": 512 * 4,
            "det_score": self.settings.det_score,
            "min_face_px": self.settings.min_face_px,
            "providers": list(self._app.models["detection"].session.get_providers()),
            "license": "NON-COMMERCIAL pretrained weights; commercial use requires a licence",
        }

    decode_image = staticmethod(FaceEngine.decode_image)

    def detect(self, image: np.ndarray) -> list[ExtractedFace]:
        with self._lock:
            faces = self._app.get(image)
        height, width = image.shape[:2]
        results: list[ExtractedFace] = []
        for face in faces:
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            x1_i = max(0, min(width - 1, int(round(x1))))
            y1_i = max(0, min(height - 1, int(round(y1))))
            x2_i = max(0, min(width, int(round(x2))))
            y2_i = max(0, min(height, int(round(y2))))
            if min(x2_i - x1_i, y2_i - y1_i) < self.settings.min_face_px:
                continue
            crop = image[y1_i:y2_i, x1_i:x2_i]
            if crop.size == 0:
                continue
            blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            vector = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm < 1e-12:
                continue
            attributes = {}
            sex = getattr(face, "sex", None)
            age = getattr(face, "age", None)
            if sex in ("M", "F"):
                attributes["sex"] = "male" if sex == "M" else "female"
            if age is not None:
                attributes["age"] = int(age)
            results.append(
                ExtractedFace(
                    vector=vector / norm,
                    bbox=(x1_i, y1_i, x2_i - x1_i, y2_i - y1_i),
                    det_score=float(face.det_score),
                    blur=blur,
                    landmarks=[(float(x), float(y)) for x, y in (face.kps if face.kps is not None else [])],
                    sex=attributes.get("sex"),
                    age=attributes.get("age"),
                    attributes=attributes,
                )
            )
        results.sort(key=lambda item: item.bbox[2] * item.bbox[3], reverse=True)
        return results

    extract = FaceEngine.extract
    extract_one = FaceEngine.extract_one
    extract_path = FaceEngine.extract_path


def create_face_engine(settings: Settings, allow_model_download: bool = True):
    if settings.face_engine.lower() == "insightface":
        return InsightFaceEngine(settings, allow_model_download=allow_model_download)
    return FaceEngine(settings, allow_model_download=allow_model_download)
