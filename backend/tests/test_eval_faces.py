"""Тесты расчёта FAR/FRR/EER на данных с известным ответом (без камеры и без InsightFace)."""
import numpy as np

from smartpos.eval_faces import (
    Sample, bootstrap_eer, by_condition, eer, evaluate, far_frr, identification, ident_rates,
    list_images, pair_scores, threshold_at_far,
)


def test_far_frr_known_values():
    gen = np.array([0.9, 0.8, 0.7, 0.4])
    imp = np.array([0.5, 0.2, 0.1, 0.0])
    far, frr = far_frr(gen, imp, np.array([0.45, 0.0, 1.0]))
    assert far[0] == 0.25 and frr[0] == 0.25        # порог 0.45: один impostor >= t, одна genuine < t
    assert far[1] == 1.0 and frr[1] == 0.0          # порог 0: принимаем всех
    assert far[2] == 0.0 and frr[2] == 1.0          # порог 1: не принимаем никого


def test_eer_perfectly_separated_is_zero():
    e, t = eer(np.array([0.8, 0.9, 0.85]), np.array([0.1, 0.2, 0.0]))
    assert e == 0.0 and 0.2 < t <= 0.8


def test_eer_identical_distributions_is_about_half():
    rng = np.random.default_rng(0)
    e, _ = eer(rng.normal(size=4000), rng.normal(size=4000))
    assert 0.45 < e < 0.55


def test_threshold_at_far_and_measurability():
    rng = np.random.default_rng(1)
    gen, imp = rng.normal(0.8, 0.05, 500), rng.normal(0.1, 0.1, 500)
    r = threshold_at_far(gen, imp, 0.01)
    assert r["far"] <= 0.01 and r["measurable"] is True                   # 500 >= 3/0.01 = 300
    assert threshold_at_far(gen, imp, 0.001)["measurable"] is False       # 500 < 3000
    assert threshold_at_far(gen, np.tile(imp, 10), 0.001)["measurable"] is True


def _people(n_persons=10, noise=0.35, seed=0, conds=("base", "dim", "mask")):
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_persons):
        centre = rng.normal(size=512)
        for c in conds:
            v = centre + rng.normal(scale=noise, size=512)  # норма центра ~22.6; сходство снимков ~ 1/(1+noise^2)
            out.append(Sample(f"p{p:02d}", c, f"p{p}/{c}.jpg", (v / np.linalg.norm(v)).astype("float32")))
    return out


def test_pair_counts():
    s = _people(5)
    gen, imp = pair_scores(s)
    assert sum(len(v) for v in gen.values()) == 5 * 3          # по 3 пары на человека (3 снимка)
    assert sum(len(v) for v in imp.values()) == 10 * 9         # C(5,2) пар людей * 3*3 снимков


def test_missing_faces_are_counted_not_scored():
    s = _people(5)
    s[0].vec = None
    res = evaluate(s, threshold=0.45)
    assert abs(res["fta_overall"] - 1 / 15) < 1e-9
    assert res["images"] == 15


def test_by_condition_frr_rises_with_a_hard_condition():
    rng = np.random.default_rng(3)
    out = []
    for p in range(12):
        centre = rng.normal(size=512)
        for c, noise in (("base", 0.35), ("easy", 0.35), ("mask", 3.0)):
            v = centre + rng.normal(scale=noise, size=512)
            out.append(Sample(f"p{p}", c, f"{p}{c}", (v / np.linalg.norm(v)).astype("float32")))
    rows = {r["condition"]: r for r in by_condition(out, threshold=0.45)}
    assert rows["mask"]["frr"] > rows["easy"]["frr"]
    assert rows["mask"]["mean_score"] < rows["easy"]["mean_score"]


def test_identification_separates_enrolled_and_strangers():
    s = _people(12, seed=5)
    ident = identification(s, enroll_frac=0.5, repeats=20)
    assert len(ident["g_score"]) and len(ident["i_score"])
    good = ident_rates(ident, 0.5)
    # на пороге 0.999 никого не узнают, на пороге -1 принимают всех незнакомцев
    assert ident_rates(ident, 0.999)["DIR"] == 0.0
    assert ident_rates(ident, -1.0)["FPIR"] == 1.0
    assert good["DIR"] > good["FPIR"]  # при нормальном пороге своих узнают чаще, чем чужих принимают


def test_bootstrap_interval_contains_point_estimate():
    s = _people(15, seed=7)
    gen, imp = pair_scores(s)
    g, i = np.concatenate(list(gen.values())), np.concatenate(list(imp.values()))
    point = eer(g, i)[0]
    lo, hi = bootstrap_eer(gen, imp, n_boot=100)
    assert lo <= hi and lo - 0.05 <= point <= hi + 0.05


def test_dataset_layout_parsing(tmp_path):
    for rel in ("s01/base__1.jpg", "s01/dim__2.png", "s02/photo.jpg", ".hidden/x.jpg", "s02/notes.txt"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    found = [(p, c) for p, c, _ in list_images(tmp_path)]
    assert found == [("s01", "base"), ("s01", "dim"), ("s02", "base")]
