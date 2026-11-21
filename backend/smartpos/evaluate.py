"""Офлайн-оценка рекомендательной модели: обучение на первых 80% заказов, тест на последних 20%.

Запуск:  python -m smartpos.evaluate
Сравниваем гибридную модель с двумя бейзлайнами: «самое популярное» и «популярное для этого напитка».
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from .db import connect
from .recommender import Recommender, load_orders


def main() -> dict:
    con = connect()
    orders = load_orders(con)
    food_ids = [r["id"] for r in con.execute("SELECT id FROM products WHERE category='food'")]
    rec = Recommender()
    cut = int(len(orders) * 0.8)
    rec.fit(orders, food_ids, train_frac=0.8)
    X, y, meta = rec.build_rows(orders)  # история строго из прошлого; индексы заказов в meta
    test = np.array([m[0] >= cut for m in meta])
    Xt, yt = X[test], y[test]
    mt = [m for m, t in zip(meta, test) if t]

    scores = {
        "hybrid_model": rec.model.predict_proba(Xt)[:, 1],
        "baseline_popular": Xt[:, 0] * 0 + np.array([rec.p_food[f] for _, f in mt]),
        "baseline_by_drink": Xt[:, 3],
    }
    result = {"train_orders": cut, "test_orders": len(orders) - cut, "test_rows": int(len(yt))}
    for name, s in scores.items():
        by_order = defaultdict(list)
        for (oi, f), sc, lab in zip(mt, s, yt):
            by_order[oi].append((sc, lab))
        with_food = [v for v in by_order.values() if any(l for _, l in v)]
        hit1 = np.mean([max(v, key=lambda t: t[0])[1] for v in with_food])
        result[name] = {"AUC": round(float(roc_auc_score(yt, s)), 4), "hit@1 (orders with add-on)": round(float(hit1), 4)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
