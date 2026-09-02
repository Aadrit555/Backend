"""Test suite for Custom Vision API Endpoints."""

import base64
import io
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


def _gen_img_b64(color=(0, 255, 0)):
    img = Image.new("RGB", (32, 32), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def test_api_custom_vision_flow():
    client = TestClient(app)

    # 1. Train via API
    payload = {
        "classes": {
            "Green Item": [_gen_img_b64((0, 255, 0)) for _ in range(3)],
            "Yellow Item": [_gen_img_b64((255, 255, 0)) for _ in range(3)],
        },
        "backbone": "mobilenet_v3_small",
        "epochs": 2,
        "lr": 0.005,
        "batch_size": 4,
    }

    resp = client.post("/api/classifier/train", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    model_info = data["model"]
    model_id = model_info["model_id"]
    assert model_id.startswith("cv_")

    # 2. Predict via JSON
    test_img = _gen_img_b64((0, 240, 0))
    pred_resp = client.post("/api/classifier/predict", json={"image": test_img, "model_id": model_id})
    assert pred_resp.status_code == 200, pred_resp.text
    pred_data = pred_resp.json()
    assert pred_data["status"] == "success"
    assert len(pred_data["predictions"]) == 2
    assert pred_data["speed_ms"] > 0

    # 3. Predict via legacy /api/teachable/predict endpoint
    legacy_pred = client.post("/api/teachable/predict", json={"image": test_img, "model_id": model_id})
    assert legacy_pred.status_code == 200
    assert legacy_pred.json()["status"] == "success"

    # 4. List models
    models_resp = client.get("/api/classifier/models")
    assert models_resp.status_code == 200
    models = models_resp.json()["models"]
    assert any(m["model_id"] == model_id for m in models)

    # 5. Download model checkpoint (.pth)
    dl_resp = client.get(f"/api/classifier/{model_id}/download")
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 1000

    print("API E2E TESTS PASSED!")


if __name__ == "__main__":
    test_api_custom_vision_flow()

