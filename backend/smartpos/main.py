"""FastAPI-бэкенд Smart POS: идентификация гостя, карточка, подсказки ИИ, регистрация, оплата."""
import json
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import context as ctxmod
from .db import DB_PATH, connect
from .faces import FaceIndex, SimulatedEngine, make_engine
from .recommender import Order, Recommender, daypart, load_orders

WELCOME_BONUS = 200
CASHBACK = 0.05
FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"


class State:
    rec = Recommender()
    index = FaceIndex()
    engine = make_engine()
    scans: dict[str, np.ndarray] = {}  # незарегистрированные снимки живут только в памяти


S = State()


def _ensure_seasonal_products() -> None:
    """Сезонные напитки добавляются в уже существующую БД, не затрагивая историю чеков."""
    con = connect()
    con.executemany(
        "INSERT OR IGNORE INTO products (id,name,category,price,stock,hours_left) VALUES (?,?,?,?,?,NULL)",
        [(9, "Тыквенный латте", "drink", 1800, 999), (10, "Новогодний раф", "drink", 1900, 999)],
    )
    con.commit()
    con.close()


def _train() -> None:
    _ensure_seasonal_products()
    con = connect()
    food = [r["id"] for r in con.execute("SELECT id FROM products WHERE category='food'")]
    S.rec.fit(load_orders(con), food)
    S.index.rebuild(con)
    con.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not DB_PATH.exists() or not connect().execute("SELECT 1 FROM customers").fetchone():
        from .synth import generate
        generate()
    _train()
    yield


app = FastAPI(title="Smart POS & Loyalty", lifespan=lifespan)


# ---------- вспомогательное ----------
def _products(con) -> dict[int, dict]:
    return {r["id"]: dict(r) for r in con.execute("SELECT * FROM products")}


def _history(con, cid: int) -> list[Order]:
    return [o for o in load_orders(con) if o.customer_id == cid]


def customer_view(con, cid: int) -> dict:
    c = con.execute("SELECT * FROM customers WHERE id=?", (cid,)).fetchone()
    if not c:
        raise HTTPException(404, "Гость не найден")
    hist = _history(con, cid)
    prods = _products(con)
    fav = None
    drinks = [o.drink for o in hist if o.drink]
    if drinks:
        top = max(set(drinks), key=drinks.count)
        fav = {"product_id": top, "name": prods[top]["name"], "share": drinks.count(top) / len(drinks)}
    return {
        "id": c["id"], "name": c["name"], "bonus": c["bonus"], "visits": len(hist),
        "prefs": json.loads(c["prefs"]), "favorite_drink": fav,
        "last_visit": hist[-1].ts.isoformat() if hist else None,
        "biometrics": bool(con.execute("SELECT 1 FROM embeddings WHERE customer_id=?", (cid,)).fetchone()),
    }


def _identify(vec: np.ndarray, con) -> dict:
    cid, score = S.index.search(vec)
    if cid is not None:
        return {"matched": True, "score": round(score, 3), "customer": customer_view(con, cid), "scan_id": None}
    scan_id = secrets.token_hex(8)
    S.scans[scan_id] = vec
    return {"matched": False, "score": round(score, 3), "customer": None, "scan_id": scan_id}


# ---------- API ----------
class SimScan(BaseModel):
    customer_id: int | None = None


@app.get("/api/state")
def state():
    con = connect()
    return {"engine": S.engine.name, "products": list(_products(con).values()),
            "welcome_bonus": WELCOME_BONUS, "cashback": CASHBACK}


@app.get("/api/customers")
def customers():
    con = connect()
    return [dict(r) for r in con.execute("SELECT id, name FROM customers WHERE consent=1 ORDER BY id")]


@app.post("/api/simulate-scan")
def simulate_scan(body: SimScan):
    """Демо без камеры: «снимок» существующего гостя (или незнакомца, если customer_id не задан)."""
    if not isinstance(S.engine, SimulatedEngine):
        raise HTTPException(400, "Симуляция доступна только с SMARTPOS_FACE_ENGINE=simulated")
    con = connect()
    base = None
    if body.customer_id is not None:
        row = con.execute("SELECT vec FROM embeddings WHERE customer_id=?", (body.customer_id,)).fetchone()
        base = np.frombuffer(row["vec"], dtype="float32") if row else None
    return _identify(S.engine.capture(base if base is not None else S.engine.new_person()), con)


@app.post("/api/scan")
async def scan(image: UploadFile = File(...)):
    """Реальное распознавание по кадру с веб-камеры (SMARTPOS_FACE_ENGINE=insightface)."""
    if isinstance(S.engine, SimulatedEngine):
        raise HTTPException(400, "Включите SMARTPOS_FACE_ENGINE=insightface для распознавания по фото")
    vec = S.engine.embed(await image.read())
    if vec is None:
        raise HTTPException(422, "Лицо в кадре не найдено")
    return _identify(vec, connect())


class RecRequest(BaseModel):
    customer_id: int | None = None
    drink_id: int | None = None
    hour: int | None = None


@app.post("/api/recommend")
def recommend(body: RecRequest):
    con = connect()
    prods = _products(con)
    now = datetime.now()
    hour = body.hour if body.hour is not None else now.hour
    hist = _history(con, body.customer_id) if body.customer_id else []
    drink = body.drink_id
    if drink is None and hist:
        drinks = [o.drink for o in hist if o.drink]
        drink = max(set(drinks), key=drinks.count) if drinks else None
    recs = S.rec.recommend(hist, drink, hour, prods, now)
    script = None
    if recs:
        top, name = recs[0], None
        if body.customer_id:
            name = con.execute("SELECT name FROM customers WHERE id=?", (body.customer_id,)).fetchone()["name"]
        hello = f"Здравствуйте, {name}! " if name else "Здравствуйте! "
        dr = f"Вам {prods[drink]['name'].lower()}? " if drink else ""
        tail = " Как раз скоро закончится срок годности, возьмёте со скидкой?" if top["expiring"] else ""
        script = f"{hello}{dr}К этому часто берут «{top['name']}», добавить к заказу?{tail}"
    return {"drink_id": drink, "daypart": daypart(hour), "recommendations": recs, "script": script}


class KioskReq(BaseModel):
    customer_id: int | None = None
    date: str | None = None   # для демо: подменить дату (YYYY-MM-DD)
    hour: int | None = None   # для демо: подменить час


@app.get("/api/kiosk/menu")
def kiosk_menu(date: str | None = None, hour: int | None = None):
    """Меню киоска с учётом сезона, времени суток и акций."""
    con = connect()
    ctx = ctxmod.make_context(date, hour)
    items = []
    for p in _products(con).values():
        if not ctxmod.in_season(p["id"], ctx) or (p["category"] == "food" and p["stock"] <= 0):
            continue
        items.append({"id": p["id"], "name": p["name"], "category": p["category"], "stock": p["stock"],
                      **ctxmod.price_info(p, ctx)})
    return {"context": ctx, "products": items}


@app.post("/api/kiosk/suggest")
def kiosk_suggest(body: KioskReq):
    """Приветствие и персональные подсказки напитков для гостя (или популярные для нового)."""
    con = connect()
    ctx = ctxmod.make_context(body.date, body.hour)
    name = None
    hist: list[Order] = []
    if body.customer_id:
        name = con.execute("SELECT name FROM customers WHERE id=?", (body.customer_id,)).fetchone()["name"]
        hist = _history(con, body.customer_id)
    drinks = ctxmod.suggest_drinks(hist, ctx, _products(con), S.rec.p_drink)
    return {"context": ctx, "greeting": ctxmod.greeting(name, ctx), "drinks": drinks}


class Register(BaseModel):
    scan_id: str
    name: str
    phone: str
    consent: bool


@app.post("/api/register")
def register(body: Register):
    if not body.consent:
        raise HTTPException(400, "Без согласия на обработку биометрии регистрация невозможна")
    vec = S.scans.pop(body.scan_id, None)
    if vec is None:
        raise HTTPException(404, "Снимок не найден или устарел: распознайте гостя заново")
    con = connect()
    cur = con.execute(
        "INSERT INTO customers (name,phone,consent,bonus,prefs,created_at) VALUES (?,?,1,?,?,?)",
        (body.name.strip(), body.phone.strip(), WELCOME_BONUS, "[]", datetime.now().isoformat()),
    )
    con.execute("INSERT INTO embeddings VALUES (?,?)", (cur.lastrowid, vec.astype("float32").tobytes()))
    con.commit()
    S.index.rebuild(con)
    return customer_view(con, cur.lastrowid)


@app.delete("/api/customers/{cid}/biometrics")
def forget(cid: int):
    """Отзыв согласия: удаляем вектор лица, бонусный профиль остаётся."""
    con = connect()
    con.execute("DELETE FROM embeddings WHERE customer_id=?", (cid,))
    con.execute("UPDATE customers SET consent=0 WHERE id=?", (cid,))
    con.commit()
    S.index.rebuild(con)
    return {"ok": True}


class Item(BaseModel):
    product_id: int
    qty: int = 1


class ShownRec(BaseModel):
    product_id: int
    probability: float


class Checkout(BaseModel):
    customer_id: int | None = None
    items: list[Item]
    redeem: int = 0
    shown: list[ShownRec] = []
    apply_promos: bool = False   # киоск: цены с акциями (серверный пересчёт, клиенту не доверяем)
    date: str | None = None
    hour: int | None = None


@app.post("/api/checkout")
def checkout(body: Checkout):
    if not body.items:
        raise HTTPException(400, "Пустой заказ")
    con = connect()
    prods = _products(con)
    if body.apply_promos:
        ctx = ctxmod.make_context(body.date, body.hour)
        gross = sum(ctxmod.price_info(prods[i.product_id], ctx)["price"] * i.qty for i in body.items)
    else:
        gross = sum(prods[i.product_id]["price"] * i.qty for i in body.items)
    redeem = 0
    if body.customer_id and body.redeem > 0:
        bonus = con.execute("SELECT bonus FROM customers WHERE id=?", (body.customer_id,)).fetchone()["bonus"]
        redeem = min(body.redeem, bonus, gross)
    pay = gross - redeem
    earned = int(pay * CASHBACK) if body.customer_id else 0
    now = datetime.now().isoformat()
    cur = con.execute("INSERT INTO orders (customer_id,ts,total) VALUES (?,?,?)", (body.customer_id, now, gross))
    for it in body.items:
        con.execute("INSERT INTO order_items VALUES (?,?,?)", (cur.lastrowid, it.product_id, it.qty))
        con.execute("UPDATE products SET stock = MAX(stock-?,0) WHERE id=? AND category='food'", (it.qty, it.product_id))
    bought = {i.product_id for i in body.items}
    for s in body.shown:
        con.execute("INSERT INTO rec_log (ts,customer_id,product_id,probability,accepted) VALUES (?,?,?,?,?)",
                    (now, body.customer_id, s.product_id, s.probability, int(s.product_id in bought)))
    if body.customer_id:
        con.execute("UPDATE customers SET bonus = bonus - ? + ? WHERE id=?", (redeem, earned, body.customer_id))
    con.commit()
    return {"order_id": cur.lastrowid, "gross": gross, "redeemed": redeem, "to_pay": pay, "earned": earned,
            "customer": customer_view(con, body.customer_id) if body.customer_id else None}


@app.get("/api/metrics")
def metrics():
    """Живая конверсия подсказок: доля принятых рекомендаций (для пилота)."""
    con = connect()
    r = con.execute("SELECT COUNT(*) n, COALESCE(SUM(accepted),0) a FROM rec_log").fetchone()
    return {"shown": r["n"], "accepted": r["a"], "acceptance_rate": (r["a"] / r["n"]) if r["n"] else None}


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    @app.get("/kiosk")
    def kiosk():
        """Экран для гостя (планшет у стойки)."""
        return FileResponse(FRONTEND / "kiosk.html")

    @app.get("/barista")
    def barista():
        """Экран для персонала."""
        return FileResponse(FRONTEND / "index.html")
