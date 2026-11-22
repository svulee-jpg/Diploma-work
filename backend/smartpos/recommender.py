"""Recommendation Engine: вероятность того, что гость добавит позицию к заказу.

Гибридная модель (логистическая регрессия поверх признаков):
  * персональные: доля чеков гостя с этой позицией, недавняя доля, доля в то же время суток;
  * коллаборативные: P(позиция | напиток в заказе), P(позиция | время суток) по всем гостям;
  * холодный старт: у нового гостя персональные признаки сглаживаются к общему среднему.
Бизнес-правило поверх модели: позиции с истекающим сроком годности получают ранговый буст
(модельная вероятность при этом не искажается и показывается как есть).
"""
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression

PRIOR_STRENGTH = 3.0   # «виртуальные» чеки для сглаживания персональной доли
EXPIRY_HOURS = 4.0     # меньше этого остатка срока позиция считается «истекающей»
EXPIRY_BOOST = 1.25
FEATURES = ["cust_share", "cust_recent", "cust_daypart", "p_given_drink", "p_given_daypart", "log_visits"]


def daypart(hour: int) -> str:
    return "morning" if hour < 11 else "day" if hour < 17 else "evening"


@dataclass
class Order:
    customer_id: int | None
    ts: datetime
    drink: int | None
    food: set[int] = field(default_factory=set)


def load_orders(con) -> list[Order]:
    drinks = {r["id"] for r in con.execute("SELECT id FROM products WHERE category='drink'")}
    by_id: dict[int, Order] = {}
    for r in con.execute("SELECT id, customer_id, ts FROM orders ORDER BY ts"):
        by_id[r["id"]] = Order(r["customer_id"], datetime.fromisoformat(r["ts"]), None)
    for r in con.execute("SELECT order_id, product_id FROM order_items"):
        o = by_id[r["order_id"]]
        if r["product_id"] in drinks:
            o.drink = r["product_id"]
        else:
            o.food.add(r["product_id"])
    return sorted(by_id.values(), key=lambda o: o.ts)


class Recommender:
    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.food_ids: list[int] = []
        self.p_drink: dict[int, float] = {}
        self.p_food = defaultdict(float)
        self.p_food_drink: dict[tuple[int, int], float] = {}
        self.p_food_dp: dict[tuple[str, int], float] = {}

    # ---------- обучение ----------
    def _fit_stats(self, orders: list[Order]) -> None:
        n = max(len(orders), 1)
        c_food, c_drink, c_pair = defaultdict(int), defaultdict(int), defaultdict(int)
        c_dp, c_dp_food = defaultdict(int), defaultdict(int)
        for o in orders:
            dp = daypart(o.ts.hour)
            c_dp[dp] += 1
            if o.drink:
                c_drink[o.drink] += 1
            for f in o.food:
                c_food[f] += 1
                c_dp_food[(dp, f)] += 1
                if o.drink:
                    c_pair[(o.drink, f)] += 1
        self.p_drink = {d: c / n for d, c in c_drink.items()}  # популярность напитков по сети
        self.p_food = defaultdict(float, {f: c / n for f, c in c_food.items()})
        self.p_food_drink = {k: v / max(c_drink[k[0]], 1) for k, v in c_pair.items()}
        self.p_food_dp = {k: v / max(c_dp[k[0]], 1) for k, v in c_dp_food.items()}

    def _features(self, hist: list[Order], drink: int | None, food: int, hour: int, now: datetime) -> list[float]:
        pg = self.p_food[food]
        n = len(hist)
        share = (sum(food in h.food for h in hist) + PRIOR_STRENGTH * pg) / (n + PRIOR_STRENGTH)
        recent = [h for h in hist if now - h.ts <= timedelta(days=30)]
        recent_share = (sum(food in h.food for h in recent) + PRIOR_STRENGTH * pg) / (len(recent) + PRIOR_STRENGTH)
        dp = daypart(hour)
        same = [h for h in hist if daypart(h.ts.hour) == dp]
        dp_share = (sum(food in h.food for h in same) + PRIOR_STRENGTH * pg) / (len(same) + PRIOR_STRENGTH)
        return [
            share, recent_share, dp_share,
            self.p_food_drink.get((drink, food), pg) if drink else pg,
            self.p_food_dp.get((dp, food), pg),
            math.log1p(n),
        ]

    def build_rows(self, orders: list[Order]):
        """Обучающие строки; история клиента на момент заказа включает только прошлые чеки."""
        hist: dict[int, list[Order]] = defaultdict(list)
        X, y, meta = [], [], []
        for i, o in enumerate(orders):
            if o.customer_id is None or o.drink is None:
                continue
            h = hist[o.customer_id]
            for f in self.food_ids:
                X.append(self._features(h, o.drink, f, o.ts.hour, o.ts))
                y.append(int(f in o.food))
                meta.append((i, f))
            h.append(o)
        return np.array(X), np.array(y), meta

    def fit(self, orders: list[Order], food_ids: list[int], train_frac: float = 1.0) -> list[Order]:
        """Обучается на первых train_frac заказов (по времени); возвращает тестовый «хвост»."""
        self.food_ids = food_ids
        cut = int(len(orders) * train_frac)
        self._fit_stats(orders[:cut])
        X, y, meta = self.build_rows(orders[:cut])
        self.model = LogisticRegression(max_iter=1000).fit(X, y)
        return orders[cut:]

    # ---------- применение ----------
    def recommend(self, hist: list[Order], drink: int | None, hour: int, products: dict[int, dict],
                  now: datetime | None = None, top_k: int = 3) -> list[dict]:
        now = now or datetime.now()
        cands = [f for f in self.food_ids if products[f]["stock"] > 0]
        if not cands:
            return []
        X = np.array([self._features(hist, drink, f, hour, now) for f in cands])
        probs = self.model.predict_proba(X)[:, 1]
        out = []
        for f, p, x in zip(cands, probs, X):
            hl = products[f]["hours_left"]
            expiring = hl is not None and hl <= EXPIRY_HOURS
            reasons = self._reasons(hist, f, x, drink, now, expiring, hl)
            out.append({"product_id": f, "name": products[f]["name"], "probability": float(p),
                        "expiring": expiring, "reasons": reasons,
                        "_rank": float(p) * (EXPIRY_BOOST if expiring else 1.0)})
        out.sort(key=lambda r: r["_rank"], reverse=True)
        for r in out:
            del r["_rank"]
        return out[:top_k]

    def _reasons(self, hist, food, x, drink, now, expiring, hours_left) -> list[str]:
        reasons = []
        recent = sum(food in h.food and now - h.ts <= timedelta(days=30) for h in hist)
        if recent:
            reasons.append(f"гость брал это {recent} раз(а) за последние 30 дней")
        if len(hist) >= 5 and x[2] > x[0] * 1.3:
            reasons.append("чаще берёт в это время суток")
        if drink and x[3] > max(self.p_food[food], 0.01) * 1.5:
            reasons.append("с этим напитком часто берут")
        if expiring:
            reasons.append(f"срок годности истекает (≈{hours_left:.0f} ч): меньше списаний")
        if not hist:
            reasons.append("новый гость: рекомендация по поведению похожих гостей")
        return reasons
