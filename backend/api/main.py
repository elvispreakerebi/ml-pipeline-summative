"""FastAPI backend for emotion prediction and model retraining."""

import os
import json
import shutil
import tempfile
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.prediction import load_model, predict, reload_model, CLASS_LABELS
from src.preprocessing import preprocess_uploaded_files
from src.model import retrain_model, evaluate_model, load_metrics, EMOTION_LABELS
from src.visualization import (
    plot_class_distribution,
    plot_gender_distribution,
    plot_type_distribution,
    get_spectrogram_base64,
    fig_to_base64,
    plot_mel_spectrogram,
)

app = FastAPI(
    title="Emergency Call Emotion Classifier",
    description="Classify emotions in emergency call audio recordings",
    version="1.0.0",
)

# CORS - allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://emotion-frontend.onrender.com",
        "https://emocall.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track server start time
START_TIME = time.time()

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("Model loaded successfully on startup")
    except FileNotFoundError:
        print("Warning: No model found. Train a model first.")


@app.get("/health")
async def health_check():
    """Health check with uptime and model status."""
    model_loaded = False
    try:
        load_model()
        model_loaded = True
    except Exception:
        pass

    uptime_seconds = time.time() - START_TIME
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime_human": _format_uptime(uptime_seconds),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/classes")
async def get_classes():
    """Return the list of emotion classes."""
    return {"classes": list(CLASS_LABELS.values())}


@app.get("/metrics")
async def get_metrics():
    """Return current model evaluation metrics."""
    metrics = load_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="No metrics available. Evaluate the model first.")
    return metrics


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from an uploaded WAV file.

    Accepts a WAV audio file and returns the predicted emotion class,
    confidence score, and probability distribution.
    """
    if not file.filename.lower().endswith((".wav", ".wave")):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predict(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/predict/spectrogram")
async def predict_with_spectrogram(file: UploadFile = File(...)):
    """
    Predict emotion and return mel spectrogram visualization.

    Returns prediction result plus a base64-encoded PNG of the spectrogram.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predict(tmp_path)
        spectrogram_b64 = get_spectrogram_base64(tmp_path)
        result["spectrogram"] = spectrogram_b64
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/retrain")
async def retrain(
    files: List[UploadFile] = File(...),
    labels: str = Form(...),
):
    """
    Retrain the model with new labeled audio data.

    Args:
        files: Multiple WAV files for retraining.
        labels: Comma-separated emotion labels (e.g., "angry,drunk,painful").
    """
    label_list = [l.strip() for l in labels.split(",")]
    label_to_idx = {v: k for k, v in CLASS_LABELS.items()}

    if len(label_list) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Number of labels ({len(label_list)}) must match files ({len(files)})",
        )

    # Validate labels
    for label in label_list:
        if label not in label_to_idx:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid label '{label}'. Must be one of: {list(label_to_idx.keys())}",
            )

    # Save uploaded files
    saved_paths = []
    int_labels = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for file, label in zip(files, label_list):
        save_dir = os.path.join(UPLOAD_DIR, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file.filename)

        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        saved_paths.append(save_path)
        int_labels.append(label_to_idx[label])

    # Get old metrics for comparison
    old_metrics = load_metrics() or {}

    try:
        # Preprocess uploaded files
        X_new, y_new = preprocess_uploaded_files(saved_paths, int_labels)

        if len(X_new) == 0:
            raise HTTPException(status_code=400, detail="No valid audio files could be processed")

        # Retrain
        model, new_metrics = retrain_model(X_new, y_new)

        # Reload cached model
        reload_model()

        return {
            "status": "success",
            "files_processed": len(saved_paths),
            "old_metrics": {
                "accuracy": old_metrics.get("accuracy"),
                "f1_score": old_metrics.get("f1_score"),
                "precision": old_metrics.get("precision"),
                "recall": old_metrics.get("recall"),
            },
            "new_metrics": {
                "accuracy": new_metrics.get("accuracy"),
                "f1_score": new_metrics.get("f1_score"),
                "precision": new_metrics.get("precision"),
                "recall": new_metrics.get("recall"),
            },
            "retrained_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/insights/class-distribution")
async def insights_class_distribution():
    """Return class distribution data for charts."""
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata not found. Run download_data.py first.")

    df = pd.read_csv(METADATA_PATH)
    counts = df["emotion"].value_counts().to_dict()
    return {"distribution": counts, "total": len(df)}


@app.get("/insights/gender-distribution")
async def insights_gender_distribution():
    """Return gender breakdown per emotion."""
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata not found.")

    df = pd.read_csv(METADATA_PATH)
    ct = pd.crosstab(df["emotion"], df["gender"])
    return {"distribution": ct.to_dict()}


@app.get("/insights/type-distribution")
async def insights_type_distribution():
    """Return natural vs synthetic distribution."""
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata not found.")

    df = pd.read_csv(METADATA_PATH)
    counts = df["type"].value_counts().to_dict()
    return {"distribution": counts}


@app.get("/insights/sample-spectrograms")
async def insights_sample_spectrograms():
    """Return one sample spectrogram per emotion class as base64 PNG."""
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata not found.")

    df = pd.read_csv(METADATA_PATH)
    spectrograms = {}

    for emotion in EMOTION_LABELS:
        sample = df[df["emotion"] == emotion].iloc[0]
        file_path = os.path.join(BASE_DIR, sample["filepath"])
        if os.path.exists(file_path):
            spectrograms[emotion] = get_spectrogram_base64(file_path)

    return {"spectrograms": spectrograms}


def _format_uptime(seconds):
    """Format seconds into human-readable uptime string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
