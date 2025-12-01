import pytest
import pandas as pd
from datetime import datetime
from src.inference import predict_price, batch_predict
from src.schemas import HousePredictionRequest

# Example input
@pytest.fixture
def sample_request():
    return HousePredictionRequest(
        year_built=2000,
        bedrooms=3,
        bathrooms=2,
        sqft=1500,
        location="Downtown",
        condition="Good"
    )

def test_predict_price_single(sample_request):
    response = predict_price(sample_request)
    
    assert response.predicted_price > 0
    assert isinstance(response.prediction_time, str)
    assert len(response.confidence_interval) == 2
    assert response.confidence_interval[0] <= response.predicted_price <= response.confidence_interval[1]

def test_batch_predict(sample_request):
    requests = [sample_request, sample_request]
    responses = batch_predict(requests)
    
    assert len(responses) == 2
    for r in responses:
        assert r.predicted_price > 0
        assert isinstance(r.prediction_time, str)
