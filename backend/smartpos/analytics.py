"""Аналитика для дашборда управляющего: продажи, подсказки ИИ, операции, списания, лояльность, распознавание.

Раздел «инсайты» превращает цифры в рекомендации управляющему: это и есть часть СППР
(система не только показывает данные, но и предлагает действия).
"""
from datetime import datetime, time, timedelta

def _m(n) -> str:
    return f"{int(n):,}".replace(",", " ")


RISK_HOURS = 6.0   # выпечка с остатком срока годности не больше этого считается «под угрозой списания»


def _window(now: datetime, days: int):
    start = datetime.combine(now.date(), time(0)) - timedelta(days=days - 1)
    prev_start = start - timedelta(days=days)
    end = datetime.combine(now.date(), time(0)) + timedelta(days=1)
    return start, prev_start, end


def _kpi(con, a: datetime, b: datetime) -> dict:
    a, b = a.isoformat(), b.isoformat()
    rev, n = con.execute("SELECT COALESCE(SUM(total),0), COUNT(*) FROM orders WHERE ts>=? AND ts<?", (a, b)).fetchone()
    food = con.execute(
        "SELECT COUNT(DISTINCT o.id) FROM orders o JOIN order_items oi ON oi.order_id=o.id "
        "JOIN products p ON p.id=oi.product_id WHERE p.category='food' AND o.ts>=? AND o.ts<?", (a, b)).fetchone()[0]
    ident = con.execute("SELECT COUNT(*) FROM orders WHERE customer_id IS NOT NULL AND ts>=? AND ts<?", (a, b)).fetchone()[0]
    return {"revenue": rev, "orders": n, "avg_check": round(rev / n) if n else 0,
            "food_attach": round(food / n, 3) if n else 0, "identified_share": round(ident / n, 3) if n else 0}


def dashboard(con, days: int = 7, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    days = max(1, min(days, 180))
    start, prev_start, end = _window(now, days)
    a, b = start.isoformat(), end.isoformat()
    kpi, prev = _kpi(con, start, end), _kpi(con, prev_start, start)

    by_day = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT substr(ts,1,10), COALESCE(SUM(total),0), COUNT(*) FROM orders WHERE ts>=? AND ts<? GROUP BY 1", (a, b))}
    series = []
    for i in range(days):
        d = (start + timedelta(days=i)).date().isoformat()
        series.append({"date": d, "revenue": by_day.get(d, (0, 0))[0], "orders": by_day.get(d, (0, 0))[1]})

    hours = {int(r[0]): r[1] for r in con.execute(
        "SELECT CAST(substr(ts,12,2) AS INTEGER), COUNT(*) FROM orders WHERE ts>=? AND ts<? GROUP BY 1", (a, b))}
    by_hour = [{"hour": h, "orders": hours.get(h, 0)} for h in range(24)]

    top = [dict(r) for r in con.execute(
        "SELECT p.name, SUM(oi.qty) AS qty, SUM(oi.qty*p.price) AS revenue FROM order_items oi "
        "JOIN products p ON p.id=oi.product_id JOIN orders o ON o.id=oi.order_id "
        "WHERE o.ts>=? AND o.ts<? GROUP BY p.id ORDER BY qty DESC LIMIT 6", (a, b))]

    shown, accepted = con.execute("SELECT COUNT(*), COALESCE(SUM(accepted),0) FROM rec_log WHERE ts>=? AND ts<?", (a, b)).fetchone()
    acc_rev = con.execute("SELECT COALESCE(SUM(p.price),0) FROM rec_log r JOIN products p ON p.id=r.product_id "
                          "WHERE r.accepted=1 AND r.ts>=? AND r.ts<?", (a, b)).fetchone()[0]
    rec_by = [dict(r) for r in con.execute(
        "SELECT p.name, COUNT(*) AS shown, SUM(r.accepted) AS accepted FROM rec_log r JOIN products p ON p.id=r.product_id "
        "WHERE r.ts>=? AND r.ts<? GROUP BY p.id ORDER BY shown DESC LIMIT 6", (a, b))]
    recs = {"shown": shown, "accepted": accepted, "rate": round(accepted / shown, 3) if shown else None,
            "revenue": acc_rev, "by_product": rec_by}

    waits, preps = [], []
    for r in con.execute("SELECT ts, started_at, ready_at FROM orders WHERE ready_at IS NOT NULL AND ts>=? AND ts<?", (a, b)):
        t0, tr = datetime.fromisoformat(r["ts"]), datetime.fromisoformat(r["ready_at"])
        waits.append((tr - t0).total_seconds() / 60)
        if r["started_at"]:
            preps.append((tr - datetime.fromisoformat(r["started_at"])).total_seconds() / 60)
    waits.sort()
    ops = {"completed": len(waits),
           "avg_prep_min": round(sum(preps) / len(preps), 1) if preps else None,
           "avg_wait_min": round(sum(waits) / len(waits), 1) if waits else None,
           "p90_wait_min": round(waits[int(0.9 * (len(waits) - 1))], 1) if waits else None,
           "within_5min": round(sum(w <= 5 for w in waits) / len(waits), 3) if waits else None,
           "active_now": con.execute("SELECT COUNT(*) FROM orders WHERE status!='done'").fetchone()[0]}

    risk = [dict(r) | {"value": r["stock"] * r["price"]} for r in con.execute(
        "SELECT name, stock, hours_left, price FROM products WHERE category='food' AND stock>0 AND hours_left<=? "
        "ORDER BY hours_left", (RISK_HOURS,))]
    waste = {"items": risk, "at_risk_value": sum(r["value"] for r in risk)}

    loyalty = {"customers": con.execute("SELECT COUNT(*) FROM customers WHERE consent=1").fetchone()[0],
               "new": con.execute("SELECT COUNT(*) FROM customers WHERE created_at>=? AND created_at<?", (a, b)).fetchone()[0],
               "bonus_liability": con.execute("SELECT COALESCE(SUM(bonus),0) FROM customers").fetchone()[0]}

    scans, matched = con.execute("SELECT COUNT(*), COALESCE(SUM(matched),0) FROM scan_log WHERE ts>=? AND ts<?", (a, b)).fetchone()
    face = {"scans": scans, "recognized": matched, "unknown": scans - matched,
            "rate": round(matched / scans, 3) if scans else None}

    out = {"days": days, "from": start.date().isoformat(), "kpi": kpi, "prev": prev, "by_day": series, "by_hour": by_hour,
           "top_products": top, "recs": recs, "ops": ops, "waste": waste, "loyalty": loyalty, "face": face}
    out["insights"] = insights(out)
    return out


def insights(d: dict) -> list[dict]:
    """Правила, превращающие метрики в действия для управляющего."""
    out = []
    w = d["waste"]
    if w["at_risk_value"] > 0:
        names = ", ".join(i["name"] for i in w["items"][:3])
        out.append({"level": "warn", "text": f"Под угрозой списания выпечка на {_m(w['at_risk_value'])} ₸ ({names}). "
                    "Киоск уже предлагает скидку −25%; можно усилить подсказку бариста и вынести позицию на витрину."})
    busy = max(d["by_hour"], key=lambda h: h["orders"])
    if busy["orders"] >= 5:
        out.append({"level": "info", "text": f"Пик заказов в {busy['hour']:02d}:00 ({busy['orders']} за период). "
                    "Проверьте, что в этот час на смене достаточно бариста, а выпечка выставлена заранее."})
    o = d["ops"]
    if o["avg_wait_min"] is not None and o["completed"] >= 5:
        if o["avg_wait_min"] > 6 or (o["within_5min"] or 1) < 0.6:
            out.append({"level": "bad", "text": f"Среднее ожидание {o['avg_wait_min']} мин, в пределах 5 минут отдаётся "
                        f"{round((o['within_5min'] or 0) * 100)}% заказов. Рассмотрите второго бариста в пиковые часы."})
        else:
            out.append({"level": "ok", "text": f"Скорость обслуживания в норме: среднее ожидание {o['avg_wait_min']} мин."})
    r = d["recs"]
    if r["shown"] >= 20 and r["rate"] is not None:
        if r["rate"] < 0.2:
            out.append({"level": "warn", "text": f"Принимается только {round(r['rate'] * 100)}% подсказок. Стоит пересмотреть "
                        "формулировки и набор предлагаемых позиций."})
        else:
            out.append({"level": "ok", "text": f"Гости принимают {round(r['rate'] * 100)}% подсказок ИИ: "
                        f"это ≈ {_m(r['revenue'])} ₸ дополнительной выручки за период."})
    k = d["kpi"]
    if k["orders"] >= 10 and k["food_attach"] < 0.3:
        out.append({"level": "info", "text": f"Выпечка есть лишь в {round(k['food_attach'] * 100)}% заказов: "
                    "есть потенциал для допродаж (комбо «напиток + выпечка»)."})
    f = d["face"]
    if f["scans"] >= 10 and f["rate"] is not None and f["rate"] < 0.4:
        out.append({"level": "info", "text": f"Лишь {round(f['rate'] * 100)}% подходов закончились узнаванием: "
                    "предлагайте новым гостям карту лояльности с бонусом за регистрацию."})
    return out
