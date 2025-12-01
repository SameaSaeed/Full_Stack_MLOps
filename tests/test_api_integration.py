import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure src/ is in the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from src.app import app import app  # Now Python can find app.py
from schemas import HousePredictionRequest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "year_built": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1500,
        "location": "Downtown",
        "condition": "Good"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
