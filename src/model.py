"""Keras model loading, image preprocessing, and inference."""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

MODEL_PATH = os.getenv("MODEL_PATH", "models/my_classifier.h5")
IMAGE_SIZE = (64, 64)
CLASS_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
_MODEL: Any = None
_MODEL_LOCK = threading.Lock()


def get_model():
    """Load the Keras model once and reuse it for subsequent requests."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                import tensorflow as tf

                model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
                if not model_path.is_file():
                    raise FileNotFoundError(f"Keras model not found: {model_path}. Run scripts/create_model.py first.")
                _MODEL = tf.keras.models.load_model(str(model_path), compile=False)
    return _MODEL


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode, RGB-convert, resize, normalize, and batch an image."""
    if not image_bytes:
        raise ValueError("Uploaded image is empty")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Uploaded file is not a valid image") from exc
    return np.expand_dims(array, axis=0)


def predict(image_bytes: bytes) -> dict:
    """Return the top class and normalized class probabilities."""
    batch = preprocess_image(image_bytes)
    model = get_model()
    raw = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] == 0:
        raise RuntimeError("Model returned an unexpected prediction shape")
    scores = raw[0]
    # Convert arbitrary model scores to probabilities if needed.
    if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=1e-3):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / exp_scores.sum()
    else:
        scores = scores / scores.sum()
    index = int(np.argmax(scores))
    label = CLASS_LABELS[index] if index < len(CLASS_LABELS) else f"class_{index}"
    return {"class_label": label, "probabilities": [float(x) for x in scores]}
