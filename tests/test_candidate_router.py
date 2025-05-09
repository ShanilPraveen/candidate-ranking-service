from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ranking_endpoint_with_real_ids():
    payload = {
        "resumeID": "RES001",
        "vacancyID": "VAC001"
    }
    response = client.post("/candidates/ranking", json=payload)
    print("Response status:", response.status_code)
    print("Response data:", response.json())
    assert response.status_code == 200
