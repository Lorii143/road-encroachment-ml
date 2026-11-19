from fastapi.testclient import TestClient
from api.main import app, load_model


def test_health_and_prediction():
    load_model()
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    payload = {
        "distance_to_road": 12.0,
        "encroachment_depth": 38.0,
        "encroachment_area": 120.0,
        "building_area": 180.0,
        "encroachment_percentage": 66.7,
        "building_type": "commercial",
        "latitude": -1.2864,
        "longitude": 36.8792,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_severity"] in {"Low", "Medium", "High", "Critical"}
