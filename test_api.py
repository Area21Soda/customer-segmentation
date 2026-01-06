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
    assert "cluster" in response.get_json()
