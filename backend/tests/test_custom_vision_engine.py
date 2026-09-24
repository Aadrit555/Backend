"""Test suite for Native In-House Custom Vision Engine."""

import base64
import io
from PIL import Image
import pytest

from backend.custom_vision_engine import (
    train_classifier,
    predict_classification,
    list_models,
    get_models_dir,
)


def _generate_synthetic_image_b64(color: tuple[int, int, int]) -> str:
    img = Image.new("RGB", (64, 64), color=color)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_custom_vision_e2e_training_and_inference():
    # 1. Create sample synthetic datasets (Red vs Blue)
    red_samples = [_generate_synthetic_image_b64((255, 0, 0)) for _ in range(4)]
    blue_samples = [_generate_synthetic_image_b64((0, 0, 255)) for _ in range(4)]

    classes_data = {
        "Red Object": red_samples,
        "Blue Object": blue_samples,
    }

    # 2. Train with mobilenet_v3_small
    meta = train_classifier(
        classes_data=classes_data,
        backbone="mobilenet_v3_small",
        epochs=3,
        lr=0.005,
        batch_size=4,
    )

    assert meta["model_id"].startswith("cv_")
    assert meta["backbone"] == "mobilenet_v3_small"
    assert meta["total_samples"] == 8
    assert meta["top1_accuracy"] >= 0.5
    assert meta["fit_time_seconds"] > 0

    # 3. Verify file artifacts exist
    model_id = meta["model_id"]
    model_dir = get_models_dir() / model_id
    assert (model_dir / "model.pth").exists()
    assert (model_dir / "metadata.json").exists()

    # 4. Verify model listing
    models = list_models()
    matching = [m for m in models if m["model_id"] == model_id]
    assert len(matching) == 1

    # 5. Run inference on Red image
    red_test = _generate_synthetic_image_b64((240, 10, 10))
    res_red = predict_classification(red_test, model_id=model_id)

    assert res_red["status"] == "success"
    assert res_red["model_id"] == model_id
    assert len(res_red["predictions"]) == 2
    assert "speed_ms" in res_red
    print(f"Inference latency: {res_red['speed_ms']} ms")
    print(f"Predictions: {res_red['predictions']}")


if __name__ == "__main__":
    test_custom_vision_e2e_training_and_inference()
    print("ALL TESTS PASSED!")

