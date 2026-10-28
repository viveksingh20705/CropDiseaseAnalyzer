"""Load the trained crop-disease model and return Top-1 / Top-3 predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from PIL import Image

from tf_keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "crop_disease_model.keras"
CLASSES_PATH = MODELS_DIR / "classes.json"
IMAGE_SIZE = (224, 224)

_model = None
_class_names: list[str] | None = None


def load_artifacts():
    global _model, _class_names
    if _model is not None and _class_names is not None:
        return _model, _class_names
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class map not found at {CLASSES_PATH}. Run `python train.py` first."
        )
    _model = load_model(MODEL_PATH, compile=False)
    _class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return _model, _class_names


def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def _format_prediction(label: str, probability: float) -> dict:
    return {
        "label": label,
        "confidence": round(float(probability) * 100.0, 2),
    }


def predict_crop_disease(image_path: str) -> dict:
    model, class_names = load_artifacts()
    batch = preprocess_image(image_path)
    probabilities = model.predict(batch, verbose=0)[0]
    ranked = np.argsort(probabilities)[::-1]
    top3_idx = ranked[: min(3, len(class_names))]
    top3 = [_format_prediction(class_names[i], probabilities[i]) for i in top3_idx]
    return {
        "top1": top3[0],
        "top3": top3,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict crop disease from a leaf image.")
    parser.add_argument("image_path", help="Path to a leaf image")
    args = parser.parse_args()
    result = predict_crop_disease(args.image_path)
    print(json.dumps(result, indent=2))
