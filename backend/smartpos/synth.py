"""Генератор синтетических данных: клиенты, меню, ~6 месяцев чеков, эмбеддинги.

Реальных данных Espresso Day / Luna Coffee на старте нет, поэтому поведение гостей
задаётся явными правилами (архетипы + популяционные паттерны «кофе -> выпечка»).
Это нужно честно описать в дипломе как ограничение: метрики модели на синтетике
показывают работоспособность пайплайна, а не реальный прирост продаж.

Запуск:  python -m smartpos.synth
"""
import json
from datetime import datetime, timedelta

import numpy as np

from .db import DB_PATH, connect
from .faces import identity_vector

DRINKS = [  # id, name, price
    (1, "Эспрессо", 900), (2, "Американо", 1000), (3, "Капучино 300 мл", 1400),
    (4, "Латте", 1500), (5, "Флэт уайт", 1500), (6, "Раф", 1700),
    (7, "Матча-латте", 1800), (8, "Айс-латте", 1700),
]
FOOD = [  # id, name, price, shelf_life_h
    (11, "Круассан", 1100, 10), (12, "Морковный кекс", 1600, 24), (13, "Чизкейк", 1900, 24),
    (14, "Печенье", 800, 72), (15, "Сэндвич с индейкой", 2200, 8), (16, "Банановый хлеб", 1200, 36),
]
NAMES = [
    "Сауле", "Айгерим", "Дана", "Тимур", "Арман", "Алина", "Мадина", "Руслан", "Камила", "Ерлан",
    "Диана", "Азамат", "Жанар", "Нурлан", "Аружан", "Данияр", "Томирис", "Санжар", "Ляззат", "Бекзат",
    "Алия", "Марат", "Салтанат", "Олжас", "Меруерт", "Адиль", "Зарина", "Кайрат", "Асель", "Ильяс",
    "Назгуль", "Темирлан", "Карина", "Жандос", "Милана", "Станислав", "Виктория", "Артём", "София", "Глеб",
]
PREFS = ["Без сахара", "Миндальное молоко", "Овсяное молоко", "Без лактозы", "Дополнительный шот"]
DAYPARTS = {"morning": (7, 11), "day": (11, 17), "evening": (17, 21)}
# P(выпечка | напиток) по утрам / в другое время: популяционная «коллаборативная» закономерность
CF_RULES = {
    (5, "morning"): (11, 0.40), (3, "morning"): (11, 0.35), (4, "evening"): (13, 0.25),
    (7, "day"): (14, 0.25), (6, "day"): (13, 0.20), (2, "day"): (15, 0.18),
}


def generate(n_customers: int = 40, days: int = 180, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = connect()
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    for pid, name, price in DRINKS:
        con.execute("INSERT INTO products VALUES (?,?,?,?,?,NULL)", (pid, name, "drink", price, 999))
    # часть выпечки «истекает»: есть повод для подсказки в духе «снизить списания»
    hours_left = {11: 3.0, 12: 20.0, 13: 18.0, 14: 60.0, 15: 6.0, 16: 30.0}
    for pid, name, price, _shelf in FOOD:
        con.execute("INSERT INTO products VALUES (?,?,?,?,?,?)",
                    (pid, name, "food", price, int(rng.integers(3, 9)), hours_left[pid]))

    drink_ids = [d[0] for d in DRINKS]
    food_ids = [f[0] for f in FOOD]
    price = {d[0]: d[2] for d in DRINKS} | {f[0]: f[2] for f in FOOD}

    for i in range(n_customers):
        cid = i + 1
        name = NAMES[i % len(NAMES)]
        if i == 0:  # якорный персонаж из концепта: Сауле - капучино + морковный кекс по утрам
            fav_drink, daypart, fav_food, food_p, visits = 3, "morning", 12, 0.55, 60
            prefs = ["Без сахара"]
        else:
            fav_drink = int(rng.choice(drink_ids))
            daypart = str(rng.choice(list(DAYPARTS), p=[0.45, 0.35, 0.2]))
            fav_food = int(rng.choice(food_ids)) if rng.random() < 0.75 else None
            food_p = float(rng.uniform(0.15, 0.55))
            visits = int(rng.integers(4, 70))
            prefs = list(rng.choice(PREFS, size=int(rng.integers(0, 3)), replace=False))
        created = now - timedelta(days=days)
        con.execute(
            "INSERT INTO customers (id,name,phone,consent,bonus,prefs,created_at) VALUES (?,?,?,?,?,?,?)",
            (cid, name, f"+7700{1000000 + cid}", 1, 0, json.dumps(prefs, ensure_ascii=False), created.isoformat()),
        )
        emb = identity_vector(cid) + rng.normal(scale=0.025, size=512)  # «регистрационный» снимок
        con.execute("INSERT INTO embeddings VALUES (?,?)", (cid, (emb / np.linalg.norm(emb)).astype("float32").tobytes()))

        bonus = 0
        for _ in range(visits):
            day = now - timedelta(days=float(rng.uniform(0, days)))
            lo, hi = DAYPARTS[daypart] if rng.random() < 0.8 else DAYPARTS[str(rng.choice(list(DAYPARTS)))]
            ts = day.replace(hour=int(rng.integers(lo, hi)), minute=int(rng.integers(0, 60)))
            part = next(k for k, (a, b) in DAYPARTS.items() if a <= ts.hour < b)
            drink = fav_drink if rng.random() < 0.8 else int(rng.choice(drink_ids))
            items = [drink]
            if fav_food and rng.random() < food_p * (1.3 if part == daypart else 0.6):
                items.append(fav_food)
            rule = CF_RULES.get((drink, part))
            if rule and rule[0] not in items and rng.random() < rule[1]:
                items.append(rule[0])
            total = sum(price[p] for p in items)
            cur = con.execute("INSERT INTO orders (customer_id,ts,total) VALUES (?,?,?)",
                              (cid, ts.isoformat(), total))
            con.executemany("INSERT INTO order_items VALUES (?,?,1)", [(cur.lastrowid, p) for p in items])
            bonus += int(total * 0.05)
        con.execute("UPDATE customers SET bonus=? WHERE id=?", (bonus % 900 + 50 if i else 450, cid))
    con.commit()
    con.close()


if __name__ == "__main__":
    generate()
    print(f"Synthetic DB created: {DB_PATH}")
