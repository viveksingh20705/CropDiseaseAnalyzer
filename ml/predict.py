"""Load the trained crop-disease model and return Top-1 / Top-3 predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "crop_disease_savedmodel"
CLASSES_PATH = MODELS_DIR / "classes.json"
IMAGE_SIZE = (224, 224)

_predict_function = None
_class_names: list[str] | None = None


def load_artifacts():
    global _predict_function, _class_names
    if _predict_function is not None and _class_names is not None:
        return _predict_function, _class_names
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"SavedModel not found at {MODEL_PATH}. Run `python train.py` first."
        )
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class map not found at {CLASSES_PATH}. Run `python train.py` first."
        )
    model = tf.saved_model.load(str(MODEL_PATH))
    _predict_function = model.signatures["serving_default"]
    _class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return _predict_function, _class_names


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
    predict_function, class_names = load_artifacts()
    batch = preprocess_image(image_path)
    input_name = next(iter(predict_function.structured_input_signature[1]))
    output_name = next(iter(predict_function.structured_outputs))
    prediction = predict_function(
        **{input_name: tf.convert_to_tensor(batch)}
    )[output_name]
    probabilities = prediction.numpy()[0]
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
