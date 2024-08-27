from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_create_item():
    response = client.post("/check-grammar",
        json={"text": "Hi"},
    )
    assert response.status_code == 200
    assert response.json() == [
      {
        "original_msg": "Hello",
        "corrected_msg": "I am a",
        "error_type": "Syntax Error"
      }
    ]
