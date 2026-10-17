
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_endpoint(client):
    response = client.post(
        "/prediction",
        json={
            "leaf_spots": 1,
            "yellowing": 0,
            "leaf_curl": 0,
            "wilting": 0,
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
