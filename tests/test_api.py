def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_predict_ok(client):
    r = client.post("/predict", json={"text": "الخدمة ممتازة"})
    assert r.status_code == 200


def test_predict_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422
