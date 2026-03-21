"""Visualization utilities for EDA, model evaluation, and API responses."""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

from .preprocessing import SAMPLE_RATE, N_MELS, MAX_LEN, DURATION

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMOTION_LABELS = ["angry", "drunk", "painful", "stressful"]
EMOTION_COLORS = ["#EF4444", "#F59E0B", "#8B5CF6", "#3B82F6"]


def plot_class_distribution(metadata_or_path):
    """Bar chart of samples per emotion class."""
    if isinstance(metadata_or_path, str):
        df = pd.read_csv(metadata_or_path)
    else:
        df = metadata_or_path

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["emotion"].value_counts().reindex(EMOTION_LABELS)
    bars = ax.bar(counts.index, counts.values, color=EMOTION_COLORS)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    return fig


def plot_gender_distribution(metadata_or_path):
    """Grouped bar chart of gender breakdown per emotion."""
    if isinstance(metadata_or_path, str):
        df = pd.read_csv(metadata_or_path)
    else:
        df = metadata_or_path

    fig, ax = plt.subplots(figsize=(8, 5))
    ct = pd.crosstab(df["emotion"], df["gender"]).reindex(EMOTION_LABELS)
    ct.plot(kind="bar", ax=ax, color=["#EC4899", "#3B82F6"])
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_title("Gender Distribution per Emotion")
    ax.legend(title="Gender")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_type_distribution(metadata_or_path):
    """Pie chart of natural vs synthetic audio."""
    if isinstance(metadata_or_path, str):
        df = pd.read_csv(metadata_or_path)
    else:
        df = metadata_or_path

    fig, ax = plt.subplots(figsize=(6, 6))
    counts = df["type"].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=["#10B981", "#6366F1"], startangle=90)
    ax.set_title("Natural vs Synthetic Audio")
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(file_path):
    """Display mel spectrogram of a WAV file."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=512,
                                    x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig


def plot_waveform(file_path):
    """Display waveform of a WAV file."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color="#3B82F6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = EMOTION_LABELS

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history["loss"], label="Train Loss", color="#EF4444")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss", color="#3B82F6")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(history["accuracy"], label="Train Accuracy", color="#EF4444")
    if "val_accuracy" in history:
        ax2.plot(history["val_accuracy"], label="Val Accuracy", color="#3B82F6")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_prediction_probabilities(probabilities, labels=None):
    """Horizontal bar chart of prediction confidence per class."""
    if labels is None:
        labels = EMOTION_LABELS

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, probabilities, color=EMOTION_COLORS)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Probabilities")
    ax.set_xlim(0, 1)

    for bar, val in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center")

    plt.tight_layout()
    return fig


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def get_spectrogram_base64(file_path):
    """Generate mel spectrogram and return as base64 PNG."""
    fig = plot_mel_spectrogram(file_path)
    return fig_to_base64(fig)
