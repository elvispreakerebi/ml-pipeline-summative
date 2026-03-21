"""
Download and prepare the Speech Emotion Recognition for Emergency Calls dataset.

Usage:
    python scripts/download_data.py                  # Download + generate metadata
    python scripts/download_data.py --skip-download   # Only generate metadata (if already downloaded)
"""

import os
import sys
import argparse
import zipfile
import glob
import pandas as pd

# Paths relative to backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATASET_DIR = os.path.join(RAW_DIR, "CUSTOM_DATASET")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")

KAGGLE_DATASET = "anuvagoyal/speech-emotion-recognition-for-emergency-calls"

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


def download_dataset():
    """Download the dataset using the Kaggle CLI."""
    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        import kaggle
        print(f"Downloading dataset: {KAGGLE_DATASET}")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=RAW_DIR,
            unzip=True,
        )
        print(f"Dataset downloaded to {RAW_DIR}")
    except ImportError:
        print("Kaggle package not installed. Trying CLI...")
        exit_code = os.system(
            f'kaggle datasets download -d {KAGGLE_DATASET} -p "{RAW_DIR}" --unzip'
        )
        if exit_code != 0:
            print(
                "\nFailed to download. Please download manually from:\n"
                "https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls\n"
                f"Extract to: {DATASET_DIR}"
            )
            sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        print(
            "\nPlease download manually from:\n"
            "https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls\n"
            f"Extract to: {DATASET_DIR}"
        )
        sys.exit(1)


def parse_filename(filename):
    """
    Parse the WAV filename to extract metadata.

    Format: EmotionNum_SentenceNum_Gender_SyntheticNatural_SpeakerNum
    Example: 01_02_01_01_05.wav -> angry, sentence 2, female, natural, speaker 5
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    if len(parts) < 5:
        return None

    emotion_code = parts[0]
    sentence_code = parts[1]
    gender_code = parts[2]
    type_code = parts[3]
    speaker_num = parts[4]

    return {
        "emotion": EMOTION_MAP.get(emotion_code, "unknown"),
        "emotion_code": int(emotion_code) - 1,  # 0-indexed for model labels
        "sentence": SENTENCE_MAP.get(sentence_code, "unknown"),
        "sentence_code": int(sentence_code),
        "gender": GENDER_MAP.get(gender_code, "unknown"),
        "type": TYPE_MAP.get(type_code, "unknown"),
        "speaker": int(speaker_num),
    }


def generate_metadata():
    """Scan all WAV files and generate metadata CSV."""
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        print("Please download the dataset first (run without --skip-download)")
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

        # Store path relative to backend/
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


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the Emergency Call Emotion dataset"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, only generate metadata CSV",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset()

    generate_metadata()
    print("\nDone! You can now run preprocessing.")


if __name__ == "__main__":
    main()
