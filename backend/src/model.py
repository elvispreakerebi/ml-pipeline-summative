"""CNN model architecture, training, evaluation, and retraining."""

import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from .preprocessing import N_MELS, MAX_LEN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

NUM_CLASSES = 4
EMOTION_LABELS = ["angry", "drunk", "painful", "stressful"]


def build_model(input_shape=(N_MELS, MAX_LEN, 1), num_classes=NUM_CLASSES):
    """
    Build a CNN for audio emotion classification.

    Architecture uses BatchNorm, Dropout, L2 regularization,
    and GlobalAveragePooling for a small-dataset-friendly design.
    """
    l2_reg = regularizers.l2(0.001)

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2_reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu", kernel_regularizer=l2_reg),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val=None, y_val=None,
                epochs=100, batch_size=16, model_path=None):
    """
    Train the CNN model with callbacks for optimization.

    Args:
        X_train, y_train: Training data (features and one-hot labels).
        X_val, y_val: Validation data. If None, uses 15% of training data.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        model_path: Where to save the model. Defaults to models/emotion_classifier.h5.

    Returns:
        Tuple of (model, history_dict).
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "emotion_classifier.h5")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model = build_model(input_shape=X_train.shape[1:])
    model.summary()

    # Split validation from training if not provided
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    else:
        validation_split = 0.15

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": callbacks,
        "verbose": 1,
    }

    if validation_data:
        fit_kwargs["validation_data"] = validation_data
    else:
        fit_kwargs["validation_split"] = 0.15

    history = model.fit(X_train, y_train, **fit_kwargs)

    # Save final model
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    return model, history_dict


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with multiple metrics.

    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: Test labels (one-hot encoded).

    Returns:
        Dict with accuracy, loss, f1, precision, recall, confusion_matrix,
        classification_report, and per-class metrics.
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Loss
    loss = model.evaluate(X_test, y_test, verbose=0)[0]

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(loss),
        "f1_score": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True
        ),
        "evaluated_at": datetime.now().isoformat(),
    }

    # Save metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEvaluation Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=EMOTION_LABELS)}")

    return metrics


def retrain_model(new_X, new_y, model_path=None, epochs=30, batch_size=16):
    """
    Retrain the model on new data using the existing model as a pretrained base.

    Freezes early convolutional layers and fine-tunes later layers
    with a lower learning rate.

    Args:
        new_X: New feature data (preprocessed spectrograms).
        new_y: New labels (integer class indices, NOT one-hot).
        model_path: Path to existing model. Defaults to models/emotion_classifier.h5.
        epochs: Number of fine-tuning epochs.
        batch_size: Batch size.

    Returns:
        Tuple of (retrained model, new metrics dict).
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "emotion_classifier.h5")

    # Load pretrained model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded pretrained model from {model_path}")

    # Freeze early layers (first conv block)
    for layer in model.layers[:3]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # One-hot encode new labels
    y_cat = to_categorical(new_y, NUM_CLASSES)

    # Combine with a portion of original training data if available
    train_path = os.path.join(BASE_DIR, "data", "train")
    X_orig_path = os.path.join(train_path, "X_train.npy")
    y_orig_path = os.path.join(train_path, "y_train.npy")

    if os.path.exists(X_orig_path) and os.path.exists(y_orig_path):
        X_orig = np.load(X_orig_path)
        y_orig = np.load(y_orig_path)

        # Use 30% of original data to prevent catastrophic forgetting
        n_orig = max(1, int(len(X_orig) * 0.3))
        indices = np.random.choice(len(X_orig), n_orig, replace=False)
        X_combined = np.concatenate([X_orig[indices], new_X], axis=0)
        y_combined = np.concatenate([y_orig[indices], y_cat], axis=0)
        print(f"Combined dataset: {n_orig} original + {len(new_X)} new = {len(X_combined)} samples")
    else:
        X_combined = new_X
        y_combined = y_cat
        print(f"Training on {len(new_X)} new samples only")

    # Shuffle
    perm = np.random.permutation(len(X_combined))
    X_combined = X_combined[perm]
    y_combined = y_combined[perm]

    # Fine-tune
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history = model.fit(
        X_combined, y_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    # Save retrained model
    retrained_path = os.path.join(MODEL_DIR, "emotion_classifier.h5")
    model.save(retrained_path)
    print(f"\nRetrained model saved to {retrained_path}")

    # Unfreeze all layers for future use
    for layer in model.layers:
        layer.trainable = True

    # Evaluate on test set if available
    test_path = os.path.join(BASE_DIR, "data", "test")
    X_test_path = os.path.join(test_path, "X_test.npy")
    y_test_path = os.path.join(test_path, "y_test.npy")

    metrics = {}
    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        metrics = evaluate_model(model, X_test, y_test)
    else:
        # Return training metrics
        metrics = {
            "accuracy": float(history.history["accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
            "retrained_at": datetime.now().isoformat(),
        }

    return model, metrics


def load_saved_model(model_path=None):
    """Load a saved model from disk."""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "emotion_classifier.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    return tf.keras.models.load_model(model_path)


def load_metrics():
    """Load saved evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Train the model from prepared data
    train_dir = os.path.join(BASE_DIR, "data", "train")
    test_dir = os.path.join(BASE_DIR, "data", "test")

    X_train = np.load(os.path.join(train_dir, "X_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    X_test = np.load(os.path.join(test_dir, "X_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    model, history = train_model(X_train, y_train, X_test, y_test)
    metrics = evaluate_model(model, X_test, y_test)
