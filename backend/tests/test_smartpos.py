import os
import tempfile

os.environ["SMARTPOS_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["SMARTPOS_FACE_ENGINE"] = "simulated"

import pytest
from fastapi.testclient import TestClient

from smartpos.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_known_guest_is_recognised(client):
    r = client.post("/api/simulate-scan", json={"customer_id": 1}).json()
    assert r["matched"] and r["customer"]["name"] == "Сауле"
    assert r["customer"]["favorite_drink"]["name"] == "Капучино 300 мл"


def test_all_guests_recognised_and_strangers_rejected(client):
    ids = [c["id"] for c in client.get("/api/customers").json()]
    hits = sum(client.post("/api/simulate-scan", json={"customer_id": i}).json()["customer"]["id"] == i for i in ids)
    assert hits == len(ids)
    assert not any(client.post("/api/simulate-scan", json={}).json()["matched"] for _ in range(20))


def test_recommendations_have_probabilities(client):
    r = client.post("/api/recommend", json={"customer_id": 1, "hour": 9}).json()
    assert r["recommendations"] and all(0 <= x["probability"] <= 1 for x in r["recommendations"])
    assert r["script"].startswith("Здравствуйте, Сауле")


def test_cold_start_registration_requires_consent_and_then_recognises(client):
    scan = client.post("/api/simulate-scan", json={}).json()
    assert not scan["matched"]
    body = {"scan_id": scan["scan_id"], "name": "Тест", "phone": "+77001112233"}
    assert client.post("/api/register", json={**body, "consent": False}).status_code == 400
    c = client.post("/api/register", json={**body, "consent": True}).json()
    assert c["bonus"] == 200
    again = client.post("/api/simulate-scan", json={"customer_id": c["id"]}).json()
    assert again["matched"] and again["customer"]["id"] == c["id"]


def test_checkout_bonus_math_and_withdrawal_of_consent(client):
    start = client.post("/api/simulate-scan", json={"customer_id": 1}).json()["customer"]["bonus"]
    r = client.post("/api/checkout", json={"customer_id": 1, "items": [{"product_id": 3}, {"product_id": 12}],
                                            "redeem": 100}).json()
    assert r["gross"] == 3000 and r["redeemed"] == 100 and r["to_pay"] == 2900 and r["earned"] == 145
    assert r["customer"]["bonus"] == start - 100 + 145
    client.delete("/api/customers/1/biometrics")
    assert not client.post("/api/simulate-scan", json={"customer_id": 1}).json()["matched"]


# ---------- киоск: сезон, праздники, акции ----------
def test_context_season_holiday_and_daypart():
    from smartpos.context import make_context
    c = make_context("2026-12-14", 9)
    assert c["season"] == "winter" and c["daypart"] == "morning"
    assert c["holiday"]["name"] == "День Независимости" and c["holiday"]["days_left"] == 2
    assert make_context("2026-07-15", 14)["holiday"] is None


def test_kiosk_menu_hides_out_of_season_and_applies_promos(client):
    autumn = client.get("/api/kiosk/menu", params={"date": "2026-10-10", "hour": 9}).json()
    names = {p["name"]: p for p in autumn["products"]}
    assert "Тыквенный латте" in names and "Новогодний раф" not in names
    assert names["Капучино 300 мл"]["price"] == 1190 and names["Капучино 300 мл"]["old_price"] == 1400  # −15% утром
    winter = client.get("/api/kiosk/menu", params={"date": "2026-12-20", "hour": 14}).json()
    assert "Новогодний раф" in {p["name"] for p in winter["products"]}


def test_kiosk_suggestions_known_guest_vs_new_guest(client):
    known = client.post("/api/kiosk/suggest", json={"customer_id": 1, "date": "2026-10-10", "hour": 9}).json()
    assert known["greeting"]["hello"] == "Доброе утро, Сауле!"
    assert known["drinks"][0]["name"] == "Капучино 300 мл" and known["drinks"][0]["is_usual"]
    guest = client.post("/api/kiosk/suggest", json={"date": "2026-07-15", "hour": 14}).json()
    assert guest["greeting"]["hello"] == "Добрый день!" and len(guest["drinks"]) == 3
    assert not any(d["is_usual"] for d in guest["drinks"])
    xmas = client.post("/api/kiosk/suggest", json={"date": "2026-12-30", "hour": 19}).json()
    assert "Новогодний раф" in [d["name"] for d in xmas["drinks"]]


def test_promo_checkout_is_priced_on_server(client):
    r = client.post("/api/checkout", json={"customer_id": None, "items": [{"product_id": 3}], "apply_promos": True,
                                            "date": "2026-10-10", "hour": 9}).json()
    assert r["gross"] == 1190
