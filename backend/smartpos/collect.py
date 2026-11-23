"""Сбор снимков добровольцев для оценки FAR/FRR (docs/FACE_EVAL_PROTOCOL.md).

Это инструмент для локальной работы, а не часть киоска:
  * выключен по умолчанию; включается переменной SMARTPOS_COLLECT=1;
  * не работает в публичном демо (SMARTPOS_DEMO=1);
  * принимает запросы только с этого же компьютера (localhost), если не задано SMARTPOS_COLLECT_ALLOW_LAN=1;
  * человека задают кодом (s01, s02...), не фамилией; снимки лежат в backend/data/faces_eval/ (в git не попадают).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

# (префикс условия, подпись для добровольца, сколько снимков нужно)
CONDITIONS = [
    ("base", "Анфас, хороший свет, без очков и маски (это снимок для регистрации)", 2),
    ("dim", "Слабый свет (лампа сбоку, вечер)", 2),
    ("back", "Контровой свет (окно за спиной)", 1),
    ("side", "Поверните голову примерно на 30° (влево или вправо)", 2),
    ("glasses", "В очках (если не носите очки: снимите, если носите)", 1),
    ("mask", "Медицинская маска или шарф на нижней части лица", 1),
    ("far", "Отойдите на 1,5 м от камеры, как у стойки", 1),
]
COND_NAMES = {c for c, _, _ in CONDITIONS}
PERSON_RE = re.compile(r"^[a-z0-9]{2,12}$")
LOOPBACK = {"127.0.0.1", "::1", "localhost"}

router = APIRouter(prefix="/api/collect")
engine_provider = lambda: None  # main.py подставляет текущий движок распознавания


def data_dir() -> Path:
    return Path(os.environ.get("SMARTPOS_COLLECT_DIR") or Path(__file__).resolve().parent.parent / "data" / "faces_eval")


def enabled() -> bool:
    return os.environ.get("SMARTPOS_COLLECT") == "1" and os.environ.get("SMARTPOS_DEMO") != "1"


def guard(request: Request) -> None:
    if not enabled():
        raise HTTPException(404, "Сбор снимков выключен (SMARTPOS_COLLECT=1 включает его локально)")
    host = request.client.host if request.client else ""
    if host not in LOOPBACK and os.environ.get("SMARTPOS_COLLECT_ALLOW_LAN") != "1":
        raise HTTPException(403, "Сбор снимков доступен только с этого компьютера")


def _person(code: str) -> str:
    code = (code or "").strip().lower()
    if not PERSON_RE.match(code):
        raise HTTPException(422, "Код участника: 2–12 латинских букв и цифр, например s01 (не фамилия)")
    return code


def _files(person: str, cond: str) -> list[Path]:
    d = data_dir() / person
    return sorted(d.glob(f"{cond}__*.jpg")) if d.exists() else []


def _state(person: str) -> dict:
    return {
        "person": person,
        "conditions": [{"id": c, "label": label, "need": need, "have": len(_files(person, c))} for c, label, need in CONDITIONS],
    }


@router.get("/state")
def state(request: Request, person: str):
    guard(request)
    p = _person(person)
    eng = engine_provider()
    return {**_state(p), "face_check": bool(eng is not None and hasattr(eng, "embed"))}


@router.post("/capture")
async def capture(request: Request, person: str = Form(...), condition: str = Form(...), image: UploadFile = File(...)):
    guard(request)
    p = _person(person)
    if condition not in COND_NAMES:
        raise HTTPException(422, "Неизвестное условие съёмки")
    raw = await image.read()
    if not raw or len(raw) > 8_000_000:
        raise HTTPException(422, "Пустой или слишком большой кадр")
    d = data_dir() / p
    d.mkdir(parents=True, exist_ok=True)
    n = len(_files(p, condition)) + 1
    (d / f"{condition}__{n}.jpg").write_bytes(raw)
    eng = engine_provider()
    found = None  # None = проверка недоступна (движок simulated); снимок сохраняется всё равно, чтобы FTA был честным
    if eng is not None and hasattr(eng, "embed"):
        found = eng.embed(raw) is not None
    return {**_state(p), "saved": f"{condition}__{n}.jpg", "face_found": found}


@router.post("/undo")
def undo(request: Request, person: str = Form(...), condition: str = Form(...)):
    guard(request)
    p = _person(person)
    if condition not in COND_NAMES:
        raise HTTPException(422, "Неизвестное условие съёмки")
    files = _files(p, condition)
    if files:
        files[-1].unlink()
    return _state(p)
