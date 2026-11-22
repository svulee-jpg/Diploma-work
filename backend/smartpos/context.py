"""Контекст для киоска: время суток, сезон, праздники РК, акции и подсказки по напиткам.

Подсказка напитка = персональные привычки гостя (с затуханием по времени и временем суток)
+ контекстный приор (сезон, время суток) + популярность по сети + бонус за акцию/праздник.
Для нового гостя персональная часть равна нулю, и остаётся контекст и популярное.
"""
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta

from .recommender import daypart

DAYPART_RU = {"morning": "утро", "day": "день", "evening": "вечер"}
SEASON_RU = {"winter": "зима", "spring": "весна", "summer": "лето", "autumn": "осень"}
WEEKDAY_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
GREETING = {"morning": "Доброе утро", "day": "Добрый день", "evening": "Добрый вечер"}

# (месяц, день, название, поздравление): государственные праздники РК
HOLIDAYS = [
    (1, 1, "Новый год", "С Новым годом!"), (3, 8, "Международный женский день", "С 8 Марта!"),
    (3, 21, "Наурыз", "С Наурызом!"), (5, 1, "День единства народа", "С Праздником единства!"),
    (5, 7, "День защитника Отечества", "С Днём защитника Отечества!"), (5, 9, "День Победы", "С Днём Победы!"),
    (7, 6, "День столицы", "С Днём столицы!"), (8, 30, "День Конституции", "С Днём Конституции!"),
    (10, 25, "День Республики", "С Днём Республики!"), (12, 16, "День Независимости", "С Днём Независимости!"),
]
HOLIDAY_WINDOW_DAYS = 7  # за сколько дней до праздника начинаем «подогревать»

FESTIVE_DRINK = 10   # Новогодний раф
PUMPKIN_DRINK = 9    # Тыквенный латте
ICED_DRINK = 8       # Айс-латте
SEASONAL = {PUMPKIN_DRINK: {"autumn"}, FESTIVE_DRINK: {"winter"}}  # вне сезона не показываем


def make_context(day: str | None = None, hour: int | None = None) -> dict:
    now = datetime.now()
    d = date.fromisoformat(day) if day else now.date()
    h = hour if hour is not None else now.hour
    month = d.month
    season = "winter" if month in (12, 1, 2) else "spring" if month <= 5 else "summer" if month <= 8 else "autumn"
    holiday = None
    for m, dd, name, congrats in HOLIDAYS:
        for year in (d.year, d.year + 1):
            hd = date(year, m, dd)
            left = (hd - d).days
            if 0 <= left <= HOLIDAY_WINDOW_DAYS:
                if holiday is None or left < holiday["days_left"]:
                    holiday = {"name": name, "days_left": left, "congrats": congrats}
    return {"date": d.isoformat(), "hour": h, "daypart": daypart(h), "daypart_ru": DAYPART_RU[daypart(h)],
            "season": season, "season_ru": SEASON_RU[season], "weekday_ru": WEEKDAY_RU[d.weekday()],
            "is_weekend": d.weekday() >= 5, "holiday": holiday}


def ctx_datetime(ctx: dict) -> datetime:
    return datetime.combine(date.fromisoformat(ctx["date"]), time(ctx["hour"]))


def in_season(product_id: int, ctx: dict) -> bool:
    return product_id not in SEASONAL or ctx["season"] in SEASONAL[product_id]


def promo_for(p: dict, ctx: dict) -> tuple[int, str] | None:
    """Акция на позицию в данном контексте: (процент скидки, подпись)."""
    if p["id"] == 3 and ctx["daypart"] == "morning":
        return 15, "Утренняя акция −15%"
    if p["id"] == ICED_DRINK and ctx["season"] == "summer" and ctx["daypart"] == "day":
        return 20, "Летний happy hour −20%"
    if p["category"] == "food" and p["hours_left"] is not None and p["hours_left"] <= 4 and p["stock"] > 0:
        return 25, "Скоро истекает срок −25%"   # снижает списания
    return None


def price_info(p: dict, ctx: dict) -> dict:
    promo = promo_for(p, ctx)
    if not promo:
        return {"price": p["price"], "old_price": None, "badge": None}
    price = int(round(p["price"] * (100 - promo[0]) / 100 / 10) * 10)
    return {"price": price, "old_price": p["price"], "badge": promo[1]}


def _season_weight(drink_id: int, season: str) -> float:
    if drink_id == ICED_DRINK:
        return {"summer": 1.0, "spring": 0.6, "autumn": 0.2, "winter": 0.05}[season]
    return {"summer": 0.4, "spring": 0.7, "autumn": 1.0, "winter": 1.0}[season]


def _daypart_weight(drink_id: int, dp: str) -> float:
    if drink_id in (1, 2, 5):            # эспрессо, американо, флэт уайт: утренние
        return {"morning": 1.0, "day": 0.7, "evening": 0.4}[dp]
    if drink_id == ICED_DRINK:
        return {"morning": 0.5, "day": 1.0, "evening": 0.6}[dp]
    return {"morning": 0.8, "day": 0.8, "evening": 0.6}[dp]


def suggest_drinks(hist, ctx: dict, products: dict[int, dict], global_share: dict[int, float], top_k: int = 3) -> list[dict]:
    now = ctx_datetime(ctx)
    w, w_dp = defaultdict(float), defaultdict(float)
    for o in hist:
        if not o.drink:
            continue
        age = max((now - o.ts).days, 0)
        k = math.exp(-age / 60)
        w[o.drink] += k
        if daypart(o.ts.hour) == ctx["daypart"]:
            w_dp[o.drink] += k
    tot, tot_dp = sum(w.values()), sum(w_dp.values())
    usual = max(w, key=w.get) if tot and w[max(w, key=w.get)] / tot >= 0.3 else None

    scored = []
    for p in products.values():
        if p["category"] != "drink" or not in_season(p["id"], ctx):
            continue
        d = p["id"]
        personal = (0.7 * w[d] / tot if tot else 0) + (0.3 * w_dp[d] / tot_dp if tot_dp else 0)
        prior = (_season_weight(d, ctx["season"]) + _daypart_weight(d, ctx["daypart"])) / 2
        promo = promo_for(p, ctx)
        bonus = (0.15 if promo else 0) + (0.3 if ctx["holiday"] and d == FESTIVE_DRINK else 0)
        if d in SEASONAL:
            bonus += 0.2  # сезонная новинка
        score = 0.5 * personal + 0.2 * prior + 0.15 * global_share.get(d, 0) * 4 + bonus
        reasons = []
        if d == usual:
            reasons.append("Как обычно")
        elif personal > 0.15:
            reasons.append("Вы часто берёте")
        if d in SEASONAL:
            reasons.append("Сезонная новинка")
        if not reasons and prior >= 0.85:
            reasons.append("Подходит к погоде и времени")
        scored.append((d, score, reasons))

    scored.sort(key=lambda t: t[1], reverse=True)
    order = [t for t in scored if t[0] == usual] + [t for t in scored if t[0] != usual]
    out = []
    for d, _score, reasons in order[:top_k]:
        out.append({"product_id": d, "name": products[d]["name"], "reason": " · ".join(reasons) or "Популярный выбор",
                    "is_usual": d == usual, **price_info(products[d], ctx)})
    return out


def greeting(name: str | None, ctx: dict) -> dict:
    hello = f"{GREETING[ctx['daypart']]}, {name}!" if name else f"{GREETING[ctx['daypart']]}!"
    h = ctx["holiday"]
    if h:
        sub = h["congrats"] if h["days_left"] == 0 else f"Скоро {h['name']}: через {h['days_left']} дн."
    elif ctx["season"] == "autumn":
        sub = "Осень на дворе: пора тыквенного латте"
    elif ctx["season"] == "winter":
        sub = "Холодно на улице? Согреем кофе"
    elif ctx["season"] == "summer":
        sub = "Жарко? Холодные напитки сегодня со скидкой"
    else:
        sub = "Что закажем сегодня?"
    return {"hello": hello, "sub": sub}
