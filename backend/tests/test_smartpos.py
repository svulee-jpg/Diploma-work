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
