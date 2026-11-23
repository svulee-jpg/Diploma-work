"""Оценка качества распознавания лиц: FAR, FRR, EER, ROC и открытая идентификация (сценарий киоска).

Методика опирается на ISO/IEC 19795-1 (оценка биометрической эффективности):
  * 1:1 верификация: пары «тот же человек» (genuine) и «разные люди» (impostor), кривая FAR/FRR по порогу, EER;
  * 1:N открытая идентификация: часть людей «зарегистрирована» в галерее, остальные выступают незнакомцами
    (это и есть киоск: гость либо узнан, либо нет). Метрики: DIR (верное узнавание), FNIR, FPIR (ложное узнавание);
  * FTA (failure to acquire): доля снимков, на которых лицо не найдено;
  * доверительные интервалы бутстрэпом по людям (пары снимков одного человека зависимы, поэтому
    перевыбираются именно люди, а не пары);
  * разбивка по условиям съёмки (свет, очки, маска, ракурс).

Формат набора данных (фото в git не попадают, папка в .gitignore):
    <data>/<person_id>/<condition>__<anything>.jpg      например  s01/base__1.jpg, s01/dim__1.jpg, s01/mask__1.jpg
Условие `base` = «регистрационный» снимок при хорошем свете, анфас. Без префикса `__` условие считается `base`.

Запуск:
    python -m smartpos.eval_faces --data backend/data/faces_eval
    python -m smartpos.eval_faces --demo        # проверка конвейера на тестовом снимке (НЕ результат исследования)
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE = "base"


# ---------------------------------------------------------------- данные

@dataclass
class Sample:
    person: str
    condition: str
    path: str
    vec: np.ndarray | None  # None = лицо не найдено (FTA)


def list_images(root: Path) -> list[tuple[str, str, Path]]:
    out = []
    for pdir in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(".")):
        for f in sorted(pdir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                cond = f.stem.split("__", 1)[0] if "__" in f.stem else BASE
                out.append((pdir.name, cond, f))
    return out


def extract(root: Path, engine, use_cache: bool = True) -> list[Sample]:
    """Эмбеддинги всех снимков. Кэш лежит в папке набора (там же, где фото; в git не попадает)."""
    root = Path(root)
    cache_file = root / ".embeddings_cache.npz"
    cache: dict[str, np.ndarray | None] = {}
    if use_cache and cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        keys, vecs, ok = z["keys"], z["vecs"], z["ok"]
        cache = {str(k): (v if o else None) for k, v, o in zip(keys, vecs, ok)}
    samples: list[Sample] = []
    for person, cond, f in list_images(root):
        key = f"{f.relative_to(root)}|{f.stat().st_mtime_ns}"
        if key not in cache:
            cache[key] = engine.embed(f.read_bytes())
        samples.append(Sample(person, cond, str(f), cache[key]))
    if use_cache:
        keys = np.array(list(cache))
        vecs = np.stack([v if v is not None else np.zeros(512, "float32") for v in cache.values()]) if cache else np.zeros((0, 512), "float32")
        ok = np.array([v is not None for v in cache.values()])
        np.savez(cache_file, keys=keys, vecs=vecs, ok=ok)
    return samples


# ---------------------------------------------------------------- 1:1 верификация

def far_frr(genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray):
    """FAR(t) = доля пар разных людей со сходством >= t; FRR(t) = доля пар одного человека со сходством < t."""
    gen, imp = np.sort(genuine), np.sort(impostor)
    far = 1.0 - np.searchsorted(imp, thresholds, side="left") / max(len(imp), 1)
    frr = np.searchsorted(gen, thresholds, side="left") / max(len(gen), 1)
    return far, frr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Равная ошибка: точка, где FAR = FRR. Возвращает (EER, порог)."""
    ts = np.unique(np.concatenate([genuine, impostor]))
    far, frr = far_frr(genuine, impostor, ts)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ts[i])


def threshold_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> dict:
    """Наименьший порог, при котором FAR не больше цели, и FRR при этом пороге."""
    ts = np.unique(np.concatenate([genuine, impostor, [1.01]]))
    far, frr = far_frr(genuine, impostor, ts)
    ok = np.where(far <= target_far)[0]
    i = int(ok[0])
    n = len(impostor)
    return {
        "target_far": target_far,
        "threshold": float(ts[i]),
        "far": float(far[i]),
        "frr": float(frr[i]),
        # правило трёх: если ошибок не наблюдалось, верхняя граница ~ 3/N; цель ниже 3/N проверить нельзя
        "measurable": bool(n >= 3.0 / target_far),
        "need_impostor_pairs": int(np.ceil(3.0 / target_far)),
    }


def pair_scores(samples: list[Sample]):
    """Все пары снимков: genuine по людям и impostor по парам людей. Только снимки с найденным лицом."""
    ok = [s for s in samples if s.vec is not None]
    persons = sorted({s.person for s in ok})
    by_p = {p: np.stack([s.vec for s in ok if s.person == p]) for p in persons}
    gen = {}
    for p, m in by_p.items():
        if len(m) >= 2:
            sim = m @ m.T
            gen[p] = sim[np.triu_indices(len(m), 1)]
    imp = {}
    for i, p in enumerate(persons):
        for q in persons[i + 1:]:
            imp[(p, q)] = (by_p[p] @ by_p[q].T).ravel()
    return gen, imp


def _flat(d):
    return np.concatenate(list(d.values())) if d else np.zeros(0)


def bootstrap_eer(gen: dict, imp: dict, n_boot: int = 300, seed: int = 0):
    """95% интервал EER: перевыбор людей с возвращением (дубликаты одного человека не образуют impostor-пар)."""
    rng = np.random.default_rng(seed)
    persons = sorted(gen)
    if len(persons) < 3:
        return None
    vals = []
    for _ in range(n_boot):
        pick = list(rng.choice(persons, size=len(persons), replace=True))
        g = np.concatenate([gen[p] for p in pick])
        parts = []
        for i, p in enumerate(pick):
            for q in pick[i + 1:]:
                if p == q:
                    continue
                parts.append(imp[(p, q) if (p, q) in imp else (q, p)])
        if not parts:
            continue
        vals.append(eer(g, np.concatenate(parts))[0])
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def by_condition(samples: list[Sample], threshold: float) -> list[dict]:
    """FRR и FTA по условиям съёмки: регистрационный снимок (base) против снимка при условии."""
    conds = sorted({s.condition for s in samples})
    rows = []
    for c in conds:
        imgs = [s for s in samples if s.condition == c]
        fta = sum(s.vec is None for s in imgs) / len(imgs)
        row = {"condition": c, "images": len(imgs), "fta": fta, "genuine_pairs": 0, "frr": None, "mean_score": None}
        if c != BASE:
            scores = []
            for p in {s.person for s in imgs}:
                base = [s.vec for s in samples if s.person == p and s.condition == BASE and s.vec is not None]
                cond = [s.vec for s in imgs if s.person == p and s.vec is not None]
                for b in base:
                    for v in cond:
                        scores.append(float(b @ v))
            if scores:
                sc = np.array(scores)
                row.update(genuine_pairs=len(sc), frr=float((sc < threshold).mean()), mean_score=float(sc.mean()))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 1:N открытая идентификация (киоск)

def identification(samples: list[Sample], enroll_frac: float = 0.7, repeats: int = 50, seed: int = 0):
    """Галерея из части людей (по одному регистрационному снимку, как в киоске), остальные люди = незнакомцы.
    Возвращает сырые оценки для расчёта DIR/FNIR/FPIR при любом пороге."""
    rng = np.random.default_rng(seed)
    ok = [s for s in samples if s.vec is not None]
    persons = sorted({s.person for s in ok})
    if len(persons) < 4:
        return None
    g_score, g_correct, i_score = [], [], []
    for _ in range(repeats):
        k = min(max(2, int(round(len(persons) * enroll_frac))), len(persons) - 1)
        enrolled = list(rng.choice(persons, size=k, replace=False))
        gallery, gal_ids, used = [], [], set()
        for p in enrolled:
            allp = [s for s in ok if s.person == p]
            if len(allp) < 2:  # нужен хотя бы один снимок для проверки кроме регистрационного
                continue
            cand = [s for s in allp if s.condition == BASE] or allp
            e = cand[int(rng.integers(len(cand)))]
            gallery.append(e.vec)
            gal_ids.append(p)
            used.add(e.path)
        if len(gallery) < 2:
            continue
        G = np.stack(gallery)
        for s in ok:
            if s.path in used:
                continue
            sims = G @ s.vec
            j = int(np.argmax(sims))
            if s.person in gal_ids:
                g_score.append(float(sims[j]))
                g_correct.append(gal_ids[j] == s.person)
            else:
                i_score.append(float(sims[j]))
    return {"g_score": np.array(g_score), "g_correct": np.array(g_correct, dtype=bool), "i_score": np.array(i_score)}


def ident_rates(ident: dict, t: float) -> dict:
    gs, gc, is_ = ident["g_score"], ident["g_correct"], ident["i_score"]
    dir_ = float(((gs >= t) & gc).mean()) if len(gs) else float("nan")
    misid = float(((gs >= t) & ~gc).mean()) if len(gs) else float("nan")
    return {
        "threshold": t,
        "DIR": dir_,                              # зарегистрированного узнали верно
        "FNIR": 1.0 - dir_,                       # не узнали или узнали не того
        "wrong_person": misid,                    # назвали чужим именем зарегистрированного (худший случай для приватности)
        "FPIR": float((is_ >= t).mean()) if len(is_) else float("nan"),  # незнакомца приняли за клиента
        "genuine_probes": int(len(gs)),
        "stranger_probes": int(len(is_)),
    }


# ---------------------------------------------------------------- отчёт

def evaluate(samples: list[Sample], threshold: float = 0.45, target_far: float = 0.001, seed: int = 0) -> dict:
    gen, imp = pair_scores(samples)
    g, i = _flat(gen), _flat(imp)
    res: dict = {
        "images": len(samples),
        "persons": len({s.person for s in samples}),
        "fta_overall": float(np.mean([s.vec is None for s in samples])) if samples else None,
        "genuine_pairs": int(len(g)),
        "impostor_pairs": int(len(i)),
    }
    if len(g) == 0 or len(i) == 0:
        res["error"] = "нужно минимум 2 снимка у человека и 2 человека"
        return res
    e, e_t = eer(g, i)
    far_t, frr_t = far_frr(g, i, np.array([threshold]))
    res["eer"] = {"value": e, "threshold": e_t, "ci95": bootstrap_eer(gen, imp, seed=seed)}
    res["at_current_threshold"] = {"threshold": threshold, "far": float(far_t[0]), "frr": float(frr_t[0])}
    res["operating_points"] = [threshold_at_far(g, i, f) for f in sorted({0.01, 0.001, target_far})]
    res["scores"] = {
        "genuine_mean": float(g.mean()), "genuine_min": float(g.min()),
        "impostor_mean": float(i.mean()), "impostor_max": float(i.max()),
    }
    res["by_condition"] = by_condition(samples, threshold)
    ident = identification(samples, seed=seed)
    if ident is not None and len(ident["g_score"]) and len(ident["i_score"]):
        res["identification_at_current"] = ident_rates(ident, threshold)
        best = [p for p in res["operating_points"] if p["target_far"] == target_far][0]
        res["identification_at_recommended"] = ident_rates(ident, max(best["threshold"], e_t))
        res["_ident"] = ident
    res["_g"], res["_i"] = g, i
    return res


def plot(res: dict, out: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    g, i = res["_g"], res["_i"]
    files = []
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(i, bins=50, alpha=.6, label="разные люди (impostor)", color="#b8651b", density=True)
    ax.hist(g, bins=50, alpha=.6, label="один человек (genuine)", color="#2f6f4f", density=True)
    ax.axvline(res["at_current_threshold"]["threshold"], color="k", ls="--", lw=1, label="текущий порог")
    ax.set_xlabel("косинусное сходство"); ax.set_ylabel("плотность"); ax.legend(fontsize=8)
    fig.tight_layout(); p = out / "score_distributions.png"; fig.savefig(p, dpi=160); plt.close(fig); files.append(p.name)
    ts = np.linspace(-0.2, 1.0, 400)
    far, frr = far_frr(g, i, ts)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(ts, far, label="FAR", color="#b8651b"); axs[0].plot(ts, frr, label="FRR", color="#2f6f4f")
    axs[0].axvline(res["at_current_threshold"]["threshold"], color="k", ls="--", lw=1)
    axs[0].set_xlabel("порог"); axs[0].set_ylabel("доля ошибок"); axs[0].legend(); axs[0].set_title("FAR и FRR по порогу")
    axs[1].loglog(np.clip(far, 1e-5, 1), np.clip(frr, 1e-5, 1), color="#5b3a1e")
    axs[1].set_xlabel("FAR"); axs[1].set_ylabel("FRR"); axs[1].set_title("DET-кривая"); axs[1].grid(True, which="both", alpha=.3)
    fig.tight_layout(); p = out / "far_frr_det.png"; fig.savefig(p, dpi=160); plt.close(fig); files.append(p.name)
    return files


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}%"


def markdown(res: dict, demo: bool, figs: list[str]) -> str:
    L = []
    if demo:
        L += ["> **ДЕМО-ПРОГОН, не результат исследования.** Снимки получены из одного тестового фото и его искажений; "
              "это проверка конвейера. Реальные цифры дают только фото добровольцев.", ""]
    L += ["# Оценка распознавания лиц (FAR / FRR)", "",
          f"Людей: **{res['persons']}**, снимков: **{res['images']}** (лицо не найдено в {_pct(res['fta_overall'])}).",
          f"Пар «один человек»: {res['genuine_pairs']}, пар «разные люди»: {res['impostor_pairs']}.", ""]
    if "error" in res:
        return "\n".join(L + [f"**Ошибка:** {res['error']}"])
    e = res["eer"]
    ci = f", 95% ДИ {_pct(e['ci95'][0])}–{_pct(e['ci95'][1])}" if e["ci95"] else " (мало людей для доверительного интервала)"
    a = res["at_current_threshold"]
    L += ["## 1:1 верификация", "",
          f"- **EER:** {_pct(e['value'])}{ci} при пороге {e['threshold']:.3f}",
          f"- При текущем пороге {a['threshold']}: **FAR {_pct(a['far'])}**, **FRR {_pct(a['frr'])}**",
          f"- Сходство: один человек в среднем {res['scores']['genuine_mean']:.3f} (минимум {res['scores']['genuine_min']:.3f}); "
          f"разные люди в среднем {res['scores']['impostor_mean']:.3f} (максимум {res['scores']['impostor_max']:.3f})", "",
          "| Цель FAR | Порог | FAR | FRR | Можно ли измерить |", "|---|---|---|---|---|"]
    for p in res["operating_points"]:
        meas = "да" if p["measurable"] else f"нет, нужно ≥ {p['need_impostor_pairs']} пар разных людей"
        L.append(f"| {_pct(p['target_far'])} | {p['threshold']:.3f} | {_pct(p['far'])} | {_pct(p['frr'])} | {meas} |")
    if "identification_at_current" in res:
        L += ["", "## 1:N открытая идентификация (как в киоске)", "",
              "Галерея из 70% людей (один регистрационный снимок), остальные приходят как незнакомцы; 50 случайных разбиений.", "",
              "| Порог | DIR (узнали верно) | FNIR | Назвали чужим именем | FPIR (незнакомца приняли за клиента) |", "|---|---|---|---|---|"]
        for k in ("identification_at_current", "identification_at_recommended"):
            r = res[k]
            L.append(f"| {r['threshold']:.3f} | {_pct(r['DIR'])} | {_pct(r['FNIR'])} | {_pct(r['wrong_person'])} | {_pct(r['FPIR'])} |")
        L.append("")
        L.append("Первая строка: текущий порог; вторая: порог, подобранный под цель по FAR. "
                 "Для кофейни критичнее FPIR и «чужое имя»: приветствие чужим именем нарушает приватность.")
    L += ["", "## По условиям съёмки (регистрационный снимок `base` против снимка при условии)", "",
          "| Условие | Снимков | Лицо не найдено | Пар | Среднее сходство | FRR при текущем пороге |", "|---|---|---|---|---|---|"]
    for r in res["by_condition"]:
        ms = "—" if r["mean_score"] is None else f"{r['mean_score']:.3f}"
        L.append(f"| {r['condition']} | {r['images']} | {_pct(r['fta'])} | {r['genuine_pairs']} | {ms} | {_pct(r['frr'])} |")
    for f in figs:
        L += ["", f"![{f}]({f})"]
    L += ["", "## Как читать", "",
          "- **FAR**: доля попыток незнакомца, принятых за зарегистрированного. **FRR**: доля попыток своего, отвергнутых.",
          "- Порог выбирают под бизнес-риск: приветствовать чужим именем хуже, чем не узнать гостя, поэтому порог смещают вверх.",
          "- Интервалы посчитаны бутстрэпом по людям; при малом числе людей они широкие, это нормально и это нужно писать в тексте диплома.",
          "- Выводы о точности допустимы только для условий и состава выборки, на которых проводилась съёмка."]
    return "\n".join(L) + "\n"


def to_json(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ---------------------------------------------------------------- демо-набор для проверки конвейера

def build_demo_dataset(root: Path) -> None:
    """Из тестового снимка insightface (несколько лиц) делает мини-набор: «люди» = лица на фото,
    «условия» = искусственные искажения. Нужен только чтобы проверить весь конвейер без добровольцев."""
    import cv2
    from insightface.app import FaceAnalysis
    from insightface.data import get_image

    img = get_image("t1")
    app = FaceAnalysis(name="buffalo_l", root=os.environ.get(
        "SMARTPOS_INSIGHTFACE_ROOT", str(Path(__file__).resolve().parent.parent / "data" / "insightface")),
        providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = sorted(app.get(img), key=lambda f: f.bbox[0])
    H, W = img.shape[:2]

    def aug(c):
        h, w = c.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
        return {
            "base": c, "mirror": cv2.flip(c, 1),
            "dim": np.clip(c * 0.45, 0, 255).astype("uint8"), "bright": np.clip(c * 1.6, 0, 255).astype("uint8"),
            "blur": cv2.GaussianBlur(c, (0, 0), 3), "lowres": cv2.resize(cv2.resize(c, (w // 3, h // 3)), (w, h)),
            "rot15": cv2.warpAffine(c, M, (w, h)),
        }

    for n, f in enumerate(faces, 1):
        x1, y1, x2, y2 = f.bbox.astype(int)
        m = int(0.6 * max(x2 - x1, y2 - y1))
        crop = img[max(0, y1 - m):min(H, y2 + m), max(0, x1 - m):min(W, x2 + m)]
        d = root / f"demo{n:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for cond, im in aug(crop).items():
            cv2.imwrite(str(d / f"{cond}__1.jpg"), im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="папка набора: <data>/<person>/<condition>__N.jpg")
    ap.add_argument("--demo", action="store_true", help="проверка конвейера на тестовом снимке (не результат)")
    ap.add_argument("--threshold", type=float, default=float(os.environ.get("SMARTPOS_MATCH_THRESHOLD", "0.45")))
    ap.add_argument("--target-far", type=float, default=0.001)
    ap.add_argument("--out", help="куда писать отчёт (по умолчанию backend/data/eval_report)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.data and not a.demo:
        ap.error("укажите --data или --demo")

    from smartpos.faces import InsightFaceEngine

    tmp = None
    if a.demo:
        tmp = tempfile.TemporaryDirectory()
        data = Path(tmp.name)
        build_demo_dataset(data)
    else:
        data = Path(a.data)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent.parent / "data" / ("eval_report_demo" if a.demo else "eval_report")
    out.mkdir(parents=True, exist_ok=True)

    samples = extract(data, InsightFaceEngine(), use_cache=not a.demo)
    res = evaluate(samples, threshold=a.threshold, target_far=a.target_far, seed=a.seed)
    figs = plot(res, out) if "error" not in res else []
    (out / "report.md").write_text(markdown(res, a.demo, figs), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(to_json(res), ensure_ascii=False, indent=2), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"Отчёт: {out}")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
