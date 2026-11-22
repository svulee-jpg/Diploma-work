"""Симулятор активности киоска за последние дни: заказы, подсказки, время приготовления, попытки распознавания.

Нужен только для демонстрации дашборда, пока нет реального пилота. Все данные СИНТЕТИЧЕСКИЕ.
Запуск:  python -m smartpos.simulate_day --days 7 --per-day 45 [--offset 7 --seed 12]
Сброс всех данных (и демо, и ваших регистраций): python -m smartpos.synth
"""
import argparse
import random
from datetime import datetime, timedelta

from . import context as ctxmod
from .db import connect
from .recommender import Recommender, daypart, load_orders

HOUR_WEIGHTS = {7: 3, 8: 8, 9: 9, 10: 6, 11: 5, 12: 8, 13: 8, 14: 6, 15: 5, 16: 5, 17: 6, 18: 6, 19: 4, 20: 2}


def simulate(days: int = 7, per_day: int = 45, seed: int = 11, offset: int = 0) -> dict:
    rnd = random.Random(seed)
    con = connect()
    prods = {r["id"]: dict(r) for r in con.execute("SELECT * FROM products")}
    food_ids = [i for i, p in prods.items() if p["category"] == "food"]
    orders = load_orders(con)
    rec = Recommender()
    rec.fit(orders, food_ids)
    customers = [r["id"] for r in con.execute("SELECT id FROM customers WHERE consent=1")]
    hist_by = {}
    for o in orders:
        hist_by.setdefault(o.customer_id, []).append(o)

    now = datetime.now()
    made = 0
    for back in range(offset + days - 1, offset - 1, -1):  # offset: сколько последних дней пропустить
        day = (now - timedelta(days=back)).replace(minute=0, second=0, microsecond=0)
        hours = [h for h in HOUR_WEIGHTS if not (back == 0 and h > now.hour)]
        if not hours:
            continue
        weights = [HOUR_WEIGHTS[h] for h in hours]
        for _ in range(per_day):
            h = rnd.choices(hours, weights)[0]
            ts = day.replace(hour=h, minute=rnd.randint(0, 59), second=rnd.randint(0, 59))
            if ts > now:
                continue
            cid = rnd.choice(customers) if rnd.random() < 0.7 else None
            ctx = ctxmod.make_context(ts.date().isoformat(), h)
            hist = hist_by.get(cid, [])
            drink = ctxmod.suggest_drinks(hist, ctx, prods, rec.p_drink)[0]["product_id"] if rnd.random() < 0.8 else \
                rnd.choice([i for i, p in prods.items() if p["category"] == "drink" and ctxmod.in_season(i, ctx)])
            items = [drink]
            recs = rec.recommend(hist, drink, h, prods, ts, top_k=1)
            shown = recs[0] if recs else None
            accepted = bool(shown) and rnd.random() < min(shown["probability"] * 1.1, 0.95)
            if accepted:
                items.append(shown["product_id"])
            total = sum(prods[i]["price"] for i in items)
            # приготовление дольше в часы пик
            peak = HOUR_WEIGHTS[h] >= 8
            start = ts + timedelta(seconds=rnd.uniform(15, 120 if peak else 60))
            ready = start + timedelta(minutes=max(1.5, rnd.gauss(4.8 if peak else 3.4, 1.0)))
            done = ready + timedelta(seconds=rnd.uniform(20, 180))
            cur = con.execute(
                "INSERT INTO orders (customer_id,ts,total,status,source,started_at,ready_at,done_at) VALUES (?,?,?,'done',?,?,?,?)",
                (cid, ts.isoformat(), total, "kiosk" if rnd.random() < 0.8 else "barista",
                 start.isoformat(), ready.isoformat(), done.isoformat()))
            con.executemany("INSERT INTO order_items VALUES (?,?,1)", [(cur.lastrowid, i) for i in items])
            if shown:
                con.execute("INSERT INTO rec_log (ts,customer_id,product_id,probability,accepted) VALUES (?,?,?,?,?)",
                            (ts.isoformat(), cid, shown["product_id"], shown["probability"], int(accepted)))
            matched = cid is not None
            con.execute("INSERT INTO scan_log (ts,matched,score) VALUES (?,?,?)",
                        (ts.isoformat(), int(matched), rnd.uniform(0.72, 0.95) if matched else rnd.uniform(0.0, 0.3)))
            made += 1
    con.commit()
    return {"orders_created": made}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--per-day", type=int, default=45)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    print(simulate(a.days, a.per_day, a.seed, a.offset))
