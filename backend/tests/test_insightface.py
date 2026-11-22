"""Интеграционный тест реального движка. Пропускается, если модель buffalo_l ещё не скачана."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("SMARTPOS_DB", os.path.join(tempfile.mkdtemp(), "test_if.db"))

import pytest

pytest.importorskip("insightface")
MODEL = Path(__file__).resolve().parent.parent / "data" / "insightface" / "models" / "buffalo_l"
if not MODEL.exists():
    pytest.skip("модель buffalo_l не скачана (запустите приложение с SMARTPOS_FACE_ENGINE=insightface)",
                allow_module_level=True)

import cv2
import numpy as np
from fastapi.testclient import TestClient
from insightface.data import get_image

from smartpos.faces import InsightFaceEngine
from smartpos.main import S, app


@pytest.fixture(scope="module")
def client():
    old = S.engine
    S.engine = InsightFaceEngine()
    with TestClient(app) as c:
        yield c
    S.engine = old


def _jpeg(img):
    return ("frame.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpeg")


def test_register_real_face_then_recognise_mirrored_shot(client):
    img = get_image("t1")
    first = client.post("/api/scan", files={"image": _jpeg(img)}).json()
    assert not first["matched"] and first["scan_id"]
    cust = client.post("/api/register", json={"scan_id": first["scan_id"], "name": "Тест", "phone": "+7", "consent": True}).json()
    again = client.post("/api/scan", files={"image": _jpeg(cv2.flip(img, 1))}).json()
    assert again["matched"] and again["customer"]["id"] == cust["id"] and again["score"] > 0.8


def test_image_without_face_is_rejected(client):
    blank = np.zeros((240, 320, 3), dtype="uint8")
    assert client.post("/api/scan", files={"image": _jpeg(blank)}).status_code == 422
