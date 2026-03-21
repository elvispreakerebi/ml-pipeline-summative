"""
Prepare the Speech Emotion Recognition for Emergency Calls dataset.

Dataset: https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls

Setup:
    1. Download from Kaggle and extract CUSTOM_DATASET/ into backend/data/raw/
    2. Run: python scripts/download_data.py

This script parses WAV filenames and generates data/metadata.csv.
"""

import os
import sys
import glob
import pandas as pd

# Paths relative to backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATASET_DIR = os.path.join(RAW_DIR, "CUSTOM_DATASET")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")

EMOTION_MAP = {
    "01": "angry",
    "02": "drunk",
    "03": "painful",
    "04": "stressful",
}

SENTENCE_MAP = {
    "01": "We need an ambulance as soon as possible.",
    "02": "Someone has been lying dead on the street.",
    "03": "A neighbor of mine is shot dead.",
    "04": "This place is on fire. Please send help.",
}

GENDER_MAP = {
    "01": "female",
    "02": "male",
}

TYPE_MAP = {
    "01": "natural",
    "02": "synthetic",
}


def parse_filename(filename):
    """
    Parse WAV filename to extract metadata.

    Format: EmotionNum_SentenceNum_Gender_SyntheticNatural_SpeakerNum
    Example: 01_02_01_01_05.wav -> angry, sentence 2, female, natural, speaker 5
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    if len(parts) < 5:
        return None

    return {
        "emotion": EMOTION_MAP.get(parts[0], "unknown"),
        "emotion_code": int(parts[0]) - 1,
        "sentence": SENTENCE_MAP.get(parts[1], "unknown"),
        "sentence_code": int(parts[1]),
        "gender": GENDER_MAP.get(parts[2], "unknown"),
        "type": TYPE_MAP.get(parts[3], "unknown"),
        "speaker": int(parts[4]),
    }


def generate_metadata():
    """Scan all WAV files and generate metadata CSV."""
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        print(f"Please download from Kaggle and extract to: {DATASET_DIR}")
        sys.exit(1)

    records = []
    wav_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)

    if not wav_files:
        print(f"No WAV files found in {DATASET_DIR}")
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV files")

    for wav_path in sorted(wav_files):
        filename = os.path.basename(wav_path)
        metadata = parse_filename(filename)

        if metadata is None:
            print(f"  Skipping (unrecognized format): {filename}")
            continue

        rel_path = os.path.relpath(wav_path, BASE_DIR)
        metadata["filepath"] = rel_path
        metadata["filename"] = filename
        metadata["speaker_folder"] = os.path.basename(os.path.dirname(wav_path))
        records.append(metadata)

    df = pd.DataFrame(records)
    df.to_csv(METADATA_PATH, index=False)

    print(f"\nMetadata saved to {METADATA_PATH}")
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df["emotion"].value_counts().to_string())
    print(f"\nGender distribution:")
    print(df["gender"].value_counts().to_string())
    print(f"\nType distribution:")
    print(df["type"].value_counts().to_string())

    return df


if __name__ == "__main__":
    generate_metadata()
    print("\nDone! You can now run preprocessing.")
