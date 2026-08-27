"""Evaluate the saved Keras model on the 20% validation split of ml/dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "crop_disease_model.keras"
CLASSES_PATH = MODELS_DIR / "classes.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def main() -> None:
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Train the model first so crop_disease_model.keras and classes.json exist.")

    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    model = load_model(MODEL_PATH)

    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    validation_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        classes=class_names,
    )
    if validation_generator.samples == 0:
        raise RuntimeError("No validation images were found. Add more images per class and retrain.")

    probabilities = model.predict(validation_generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = validation_generator.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
