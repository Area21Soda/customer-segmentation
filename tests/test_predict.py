from app import app

def test_predict_endpoint():
    client = app.test_client()

    response = client.post(
        "/predict",
        json={
            "age": 30,
            "income": 60,
            "spending": 70
        }
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "cluster" in data
    assert "segmento" in data
    assert isinstance(data["cluster"], int)
