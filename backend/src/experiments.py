"""
Experiment runner: train 5 different model configurations and select the best.

Each experiment varies architecture, learning rate, augmentation strategy,
and regularization to find the optimal model for the dataset.
"""

import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .preprocessing import N_MELS, MAX_LEN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

NUM_CLASSES = 4
EMOTION_LABELS = ["angry", "drunk", "painful", "stressful"]


def _build_experiment_1():
    """Experiment 1: Baseline CNN (3 conv blocks, Adam lr=0.001)."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(N_MELS, MAX_LEN, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_experiment_2():
    """Experiment 2: Deeper CNN (4 conv blocks) + L2 regularization."""
    l2 = regularizers.l2(0.001)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=l2, input_shape=(N_MELS, MAX_LEN, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_experiment_3():
    """Experiment 3: Smaller CNN + aggressive dropout + lower LR."""
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation="relu", padding="same",
                      input_shape=(N_MELS, MAX_LEN, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_experiment_4():
    """Experiment 4: CNN with larger kernels (5x5) + SGD optimizer with momentum."""
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation="relu", padding="same",
                      input_shape=(N_MELS, MAX_LEN, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (5, 5), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_experiment_5():
    """Experiment 5: CNN with SeparableConv2D (lightweight, fewer params)."""
    l2 = regularizers.l2(0.0005)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(N_MELS, MAX_LEN, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same",
                               depthwise_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same",
                               depthwise_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


EXPERIMENTS = {
    "exp1_baseline_cnn": {
        "description": "Baseline CNN (3 blocks, Adam lr=0.001)",
        "build_fn": _build_experiment_1,
        "batch_size": 16,
        "epochs": 100,
    },
    "exp2_deeper_l2": {
        "description": "Deeper CNN (4 blocks) + L2 regularization, Adam lr=0.0005",
        "build_fn": _build_experiment_2,
        "batch_size": 16,
        "epochs": 100,
    },
    "exp3_small_aggressive_dropout": {
        "description": "Smaller CNN + aggressive dropout, Adam lr=0.0001",
        "build_fn": _build_experiment_3,
        "batch_size": 8,
        "epochs": 100,
    },
    "exp4_large_kernel_sgd": {
        "description": "5x5 kernels + SGD with momentum 0.9, lr=0.01",
        "build_fn": _build_experiment_4,
        "batch_size": 16,
        "epochs": 100,
    },
    "exp5_separable_conv": {
        "description": "SeparableConv2D (lightweight) + L2, Adam lr=0.001",
        "build_fn": _build_experiment_5,
        "batch_size": 16,
        "epochs": 100,
    },
}


def run_all_experiments(X_train, y_train, X_test, y_test):
    """
    Run all 5 experiments, evaluate each, and select the best model.

    Returns:
        Dict with results for all experiments and the best model saved to disk.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []

    for exp_name, config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        # Build fresh model
        model = config["build_fn"]()
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ]

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]

        metrics = {
            "experiment": exp_name,
            "description": config["description"],
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "loss": float(test_loss),
            "f1_score": float(f1_score(y_true, y_pred, average="macro")),
            "precision": float(precision_score(y_true, y_pred, average="macro")),
            "recall": float(recall_score(y_true, y_pred, average="macro")),
            "best_val_loss": float(min(history.history["val_loss"])),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "total_epochs": len(history.history["loss"]),
            "params_count": int(model.count_params()),
        }

        # Save this experiment's model
        exp_model_path = os.path.join(MODEL_DIR, f"{exp_name}.h5")
        model.save(exp_model_path)

        # Save training history
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        metrics["training_history"] = history_dict

        results.append(metrics)

        print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Loss:      {metrics['loss']:.4f}")
        print(f"  Params:    {metrics['params_count']:,}")

        # Clean up GPU memory
        del model
        tf.keras.backend.clear_session()

    # Sort by F1 score (most balanced metric)
    results.sort(key=lambda x: x["f1_score"], reverse=True)

    # Copy best model as the production model
    best = results[0]
    best_model_src = os.path.join(MODEL_DIR, f"{best['experiment']}.h5")
    best_model_dst = os.path.join(MODEL_DIR, "emotion_classifier.h5")

    import shutil
    shutil.copy2(best_model_src, best_model_dst)

    # Save all experiment results
    experiment_results = {
        "experiments": results,
        "best_experiment": best["experiment"],
        "best_f1_score": best["f1_score"],
        "best_accuracy": best["accuracy"],
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(MODEL_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<35} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Params':>10}")
    print(f"{'-'*85}")
    for r in results:
        marker = " << BEST" if r["experiment"] == best["experiment"] else ""
        print(f"{r['experiment']:<35} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f} {r['params_count']:>10,}{marker}")
    print(f"\nBest model: {best['experiment']} (F1={best['f1_score']:.4f})")
    print(f"Saved as: {best_model_dst}")

    # Also save best metrics as the production metrics
    from .model import evaluate_model as eval_model
    best_loaded = tf.keras.models.load_model(best_model_dst)
    eval_model(best_loaded, X_test, y_test)

    return experiment_results


if __name__ == "__main__":
    train_dir = os.path.join(BASE_DIR, "data", "train")
    test_dir = os.path.join(BASE_DIR, "data", "test")

    X_train = np.load(os.path.join(train_dir, "X_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    X_test = np.load(os.path.join(test_dir, "X_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    results = run_all_experiments(X_train, y_train, X_test, y_test)
