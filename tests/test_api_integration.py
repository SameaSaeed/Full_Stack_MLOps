import pytest
from fastapi.testclient import TestClient
from src.app import app
from src.schemas import HousePredictionRequest

client = TestClient(app)

@pytest.fixture
def sample_payload():
    return {
        "year_built": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1500,
        "location": "Downtown",
        "condition": "Good"
    }

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_endpoint(sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "confidence_interval" in data
    assert isinstance(data["predicted_price"], float)
    assert len(data["confidence_interval"]) == 2

def test_batch_predict_endpoint(sample_payload):
    response = client.post("/batch-predict", json=[sample_payload, sample_payload])
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for r in data:
        assert "predicted_price" in r
