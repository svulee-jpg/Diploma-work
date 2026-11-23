"""Сбор снимков: выключен по умолчанию, защищён, раскладывает файлы по протоколу."""
import os
import tempfile

os.environ.setdefault("SMARTPOS_DB", os.path.join(tempfile.mkdtemp(), "test_collect.db"))

import pytest
from fastapi.testclient import TestClient

from smartpos.main import app

JPG = b"\xff\xd8\xff\xe0fake-jpeg"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SMARTPOS_COLLECT_DIR", str(tmp_path))
    with TestClient(app) as c:
        yield c


def _shot(client, person="s01", cond="base"):
    return client.post("/api/collect/capture", data={"person": person, "condition": cond}, files={"image": ("f.jpg", JPG, "image/jpeg")})


def test_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("SMARTPOS_COLLECT", raising=False)
    assert _shot(client).status_code == 404
    assert client.get("/collect").status_code == 404


def test_not_available_in_public_demo(client, monkeypatch):
    monkeypatch.setenv("SMARTPOS_COLLECT", "1")
    monkeypatch.setenv("SMARTPOS_DEMO", "1")
    assert _shot(client).status_code == 404


def test_remote_hosts_rejected_unless_allowed(client, monkeypatch):
    monkeypatch.setenv("SMARTPOS_COLLECT", "1")  # TestClient приходит с хоста «testclient», не localhost
    monkeypatch.delenv("SMARTPOS_COLLECT_ALLOW_LAN", raising=False)
    assert _shot(client).status_code == 403


def test_capture_numbers_files_and_undo(client, monkeypatch, tmp_path):
    monkeypatch.setenv("SMARTPOS_COLLECT", "1")
    monkeypatch.setenv("SMARTPOS_COLLECT_ALLOW_LAN", "1")
    r1, r2 = _shot(client), _shot(client)
    assert r1.status_code == 200 and r1.json()["saved"] == "base__1.jpg" and r2.json()["saved"] == "base__2.jpg"
    assert r1.json()["face_found"] is None                      # движок simulated: проверки лица нет
    assert sorted(p.name for p in (tmp_path / "s01").iterdir()) == ["base__1.jpg", "base__2.jpg"]
    base = next(c for c in r2.json()["conditions"] if c["id"] == "base")
    assert base["have"] == 2 and base["need"] == 2
    u = client.post("/api/collect/undo", data={"person": "s01", "condition": "base"}).json()
    assert next(c for c in u["conditions"] if c["id"] == "base")["have"] == 1


def test_rejects_names_and_unknown_conditions(client, monkeypatch):
    monkeypatch.setenv("SMARTPOS_COLLECT", "1")
    monkeypatch.setenv("SMARTPOS_COLLECT_ALLOW_LAN", "1")
    assert _shot(client, person="Иванов").status_code == 422
    assert _shot(client, person="../etc").status_code == 422
    assert _shot(client, cond="selfie").status_code == 422
