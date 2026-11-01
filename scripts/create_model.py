"""Create a small, valid Keras demo classifier artifact for this project.

This is an untrained demonstration model, suitable for validating the serving
pipeline. Replace it with a genuinely trained classifier for real use.
"""
from pathlib import Path
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "my_classifier.h5")))
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.save(MODEL_PATH, include_optimizer=False)
print(f"Created demo Keras model at {MODEL_PATH}")
