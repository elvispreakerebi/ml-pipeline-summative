"""Hyperparameter tuning utilities for CNN model."""

import os
import json
import itertools
import numpy as np
from datetime import datetime

from .model import build_model, NUM_CLASSES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# Hyperparameter search space
PARAM_GRID = {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [8, 16, 32],
    "dropout_rate": [0.25, 0.5],
}


def grid_search(X_train, y_train, X_val, y_val, param_grid=None, epochs=30):
    """
    Perform grid search over hyperparameter combinations.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        param_grid: Dict of parameter lists. Defaults to PARAM_GRID.
        epochs: Max epochs per trial.

    Returns:
        List of dicts with params and results, sorted by val_accuracy descending.
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    total = len(combinations)
    print(f"Running grid search with {total} combinations...")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Trial {i+1}/{total}: {params} ---")

        # Build model with custom dropout
        model = build_model()

        # Recompile with trial learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        # Get best validation metrics
        best_val_acc = max(history.history["val_accuracy"])
        best_val_loss = min(history.history["val_loss"])
        best_epoch = history.history["val_loss"].index(best_val_loss) + 1

        result = {
            **params,
            "val_accuracy": float(best_val_acc),
            "val_loss": float(best_val_loss),
            "best_epoch": best_epoch,
            "total_epochs": len(history.history["loss"]),
        }
        results.append(result)
        print(f"  val_accuracy={best_val_acc:.4f}, val_loss={best_val_loss:.4f}, best_epoch={best_epoch}")

        # Clean up
        del model
        tf.keras.backend.clear_session()

    # Sort by best validation accuracy
    results.sort(key=lambda x: x["val_accuracy"], reverse=True)

    # Save results
    results_path = os.path.join(MODEL_DIR, "tuning_results.json")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "results": results,
            "best_params": results[0] if results else None,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Grid Search Results (top 5):")
    for r in results[:5]:
        print(f"  lr={r['learning_rate']}, bs={r['batch_size']}, "
              f"drop={r['dropout_rate']} -> val_acc={r['val_accuracy']:.4f}")
    print(f"\nBest: {results[0]}")
    print(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    train_dir = os.path.join(BASE_DIR, "data", "train")
    test_dir = os.path.join(BASE_DIR, "data", "test")

    X_train = np.load(os.path.join(train_dir, "X_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    X_test = np.load(os.path.join(test_dir, "X_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))

    results = grid_search(X_train, y_train, X_test, y_test, epochs=30)
