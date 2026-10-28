"""Train a MobileNetV2 transfer-learning classifier from ml/dataset/<class>/ images."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40


def discover_classes(dataset_dir: Path) -> list[str]:
    classes = sorted(
        [
            path.name
            for path in dataset_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
    )
    if not classes:
        raise FileNotFoundError(
            f"No class folders found in {dataset_dir}. "
            "Add images under ml/dataset/<ClassName>/ before training."
        )
    empty = [name for name in classes if not any(dataset_dir.joinpath(name).iterdir())]
    if empty:
        raise FileNotFoundError(
            "These class folders contain no images: " + ", ".join(empty)
        )
    return classes


def build_model(num_classes: int) -> Model:
    inputs = Input(shape=(224, 224, 3))
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(max(8, len(class_names) * 0.6), max(6, len(class_names) * 0.5)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Crop Disease Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_names = discover_classes(DATASET_DIR)
    num_classes = len(class_names)
    print(f"Discovered {num_classes} classes: {class_names}")

    classes_path = MODELS_DIR / "classes.json"
    classes_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved class names to {classes_path}")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        classes=class_names,
    )
    validation_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        classes=class_names,
    )

    model = build_model(num_classes)
    checkpoint_path = MODELS_DIR / "crop_disease_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(checkpoint_path)

    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    validation_generator.reset()
    probabilities = model.predict(validation_generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = validation_generator.classes
    labels = list(range(num_classes))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    (MODELS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, class_names, MODELS_DIR / "confusion_matrix.png")

    print("\nClassification report:\n")
    print(report)
    print(f"Confusion matrix saved to {MODELS_DIR / 'confusion_matrix.png'}")
    print(f"Best model saved to {checkpoint_path}")
    print(
        "Final epoch metrics — "
        f"acc={history.history['accuracy'][-1]:.4f}, "
        f"loss={history.history['loss'][-1]:.4f}, "
        f"val_acc={history.history['val_accuracy'][-1]:.4f}, "
        f"val_loss={history.history['val_loss'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
