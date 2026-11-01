from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.main import app
from src import model


@pytest.fixture
def def_client():
    # The real TensorFlow model is deliberately not loaded in API unit tests.
    with patch.object(model, "get_model", return_value=object()):
        with TestClient(app) as client:
            yield client


def make_png():
    image = Image.new("RGB", (12, 12), color=(120, 30, 200))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_health_returns_ok(def_client):
    response = def_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_image_returns_mocked_prediction(def_client):
    expected = {"class_label": "dog", "probabilities": [0.99]}
    with patch("src.main.model.predict", return_value=expected) as mocked_predict:
        response = def_client.post(
            "/predict", files={"file": ("sample.png", make_png(), "image/png")}
        )
    assert response.status_code == 200
    assert response.json() == expected
    mocked_predict.assert_called_once()


def test_predict_rejects_text_file(def_client):
    response = def_client.post(
        "/predict", files={"file": ("notes.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_rejects_corrupt_image(def_client):
    with patch("src.main.model.predict", side_effect=ValueError("Uploaded file is not a valid image")):
        response = def_client.post(
            "/predict", files={"file": ("broken.png", b"broken", "image/png")}
        )
    assert response.status_code == 422


def test_predict_returns_500_for_unexpected_model_error(def_client):
    with patch("src.main.model.predict", side_effect=RuntimeError("secret internal detail")):
        response = def_client.post(
            "/predict", files={"file": ("sample.png", make_png(), "image/png")}
        )
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
