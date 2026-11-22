"""Модуль Computer Vision: эмбеддинги лиц и поиск совпадения в векторной базе (FAISS).

Два движка с общим интерфейсом:
  * SimulatedEngine  - детерминированные «лица» для разработки и демо без камеры;
  * InsightFaceEngine - реальное распознавание (ArcFace, 512-мерные эмбеддинги).
Выбор: переменная окружения SMARTPOS_FACE_ENGINE=simulated|insightface.
"""
import os
from pathlib import Path

import faiss
import numpy as np

DIM = 512
# Косинусное сходство. На тестовом снимке (InsightFace buffalo_l) один человек даёт >= 0.93,
# разные люди <= 0.21, но для реальных условий (свет, ракурс) порог нужно калибровать (FAR/FRR).
MATCH_THRESHOLD = float(os.environ.get("SMARTPOS_MATCH_THRESHOLD", "0.45"))


def _unit(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    return v / (np.linalg.norm(v) + 1e-9)


def identity_vector(seed: int) -> np.ndarray:
    """Воспроизводимый «вектор личности» (для синтетических данных)."""
    return _unit(np.random.default_rng(10_000 + seed).normal(size=DIM))


class FaceIndex:
    """Векторная база клиентов: косинусное сходство через FAISS IndexFlatIP."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.index = faiss.IndexFlatIP(DIM)

    def rebuild(self, con) -> None:
        rows = con.execute(
            "SELECT e.customer_id, e.vec FROM embeddings e "
            "JOIN customers c ON c.id = e.customer_id WHERE c.consent = 1"
        ).fetchall()
        self.index = faiss.IndexFlatIP(DIM)
        self.ids = [r["customer_id"] for r in rows]
        if rows:
            self.index.add(np.stack([np.frombuffer(r["vec"], dtype="float32") for r in rows]))

    def search(self, vec: np.ndarray) -> tuple[int | None, float]:
        if not self.ids:
            return None, 0.0
        scores, idx = self.index.search(_unit(vec)[None, :], 1)
        score = float(scores[0][0])
        return (self.ids[int(idx[0][0])], score) if score >= MATCH_THRESHOLD else (None, score)


class SimulatedEngine:
    name = "simulated"

    def __init__(self, noise: float = 0.03, seed: int | None = None) -> None:
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def new_person(self) -> np.ndarray:
        return _unit(self.rng.normal(size=DIM))

    def capture(self, base: np.ndarray) -> np.ndarray:
        """«Снимок» человека: его базовый вектор + шум освещения/ракурса."""
        return _unit(base + self.rng.normal(scale=self.noise, size=DIM))


class InsightFaceEngine:
    name = "insightface"

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # тяжёлый импорт, поэтому ленивый

        root = os.environ.get(
            "SMARTPOS_INSIGHTFACE_ROOT",
            str(Path(__file__).resolve().parent.parent / "data" / "insightface"),
        )
        self.app = FaceAnalysis(name="buffalo_l", root=root, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed(self, image_bytes: bytes) -> np.ndarray | None:
        import cv2

        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = self.app.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return _unit(face.normed_embedding)


def make_engine():
    kind = os.environ.get("SMARTPOS_FACE_ENGINE", "simulated").lower()
    return InsightFaceEngine() if kind == "insightface" else SimulatedEngine()
