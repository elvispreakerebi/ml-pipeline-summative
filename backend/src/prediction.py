"""Prediction module: load model and classify audio emotions."""

import os
import numpy as np
import tensorflow as tf

from .preprocessing import preprocess_single_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_classifier.h5")

CLASS_LABELS = {
    0: "angry",
    1: "drunk",
    2: "painful",
    3: "stressful",
}

# Cached model singleton
_model = None
_model_path = None


def load_model(model_path=None):
    """Load and cache the Keras model."""
    global _model, _model_path

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

    # Return cached model if same path
    if _model is not None and _model_path == model_path:
        return _model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    _model = tf.keras.models.load_model(model_path)
    _model_path = model_path
    print(f"Model loaded from {model_path}")
    return _model


def reload_model():
    """Force reload the model (e.g., after retraining)."""
    global _model, _model_path
    _model = None
    _model_path = None
    return load_model()


def predict(file_path, model=None):
    """
    Predict emotion from a WAV file.

    Args:
        file_path: Path to the WAV file.
        model: Optional pre-loaded model. If None, uses cached model.

    Returns:
        Dict with class_label, confidence, and probabilities.
    """
    if model is None:
        model = load_model()

    features = preprocess_single_file(file_path)
    probabilities = model.predict(features, verbose=0)[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return {
        "class_label": CLASS_LABELS[predicted_class],
        "class_index": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_LABELS[i]: float(p) for i, p in enumerate(probabilities)
        },
    }


def predict_from_array(features, model=None):
    """Predict from pre-processed feature array."""
    if model is None:
        model = load_model()

    if features.ndim == 3:
        features = np.expand_dims(features, axis=0)

    probabilities = model.predict(features, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "class_label": CLASS_LABELS[predicted_class],
        "class_index": predicted_class,
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            CLASS_LABELS[i]: float(p) for i, p in enumerate(probabilities)
        },
    }
