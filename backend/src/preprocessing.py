"""Audio preprocessing: WAV to mel spectrogram conversion and augmentation."""

import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds (dataset clips are ~3s)
N_MELS = 128
MAX_LEN = 128  # time frames (pad/truncate to this)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_extract_features(file_path, sr=SAMPLE_RATE, n_mels=N_MELS, max_len=MAX_LEN):
    """
    Load a WAV file and convert to a log-mel spectrogram.

    Args:
        file_path: Path to the WAV file.
        sr: Target sample rate.
        n_mels: Number of mel frequency bands.
        max_len: Number of time frames to pad/truncate to.

    Returns:
        np.ndarray of shape (n_mels, max_len, 1) — ready for CNN input.
    """
    # Load audio, resample to target sr, clip to DURATION
    audio, _ = librosa.load(file_path, sr=sr, duration=DURATION)

    # Pad if shorter than expected duration
    expected_len = int(sr * DURATION)
    if len(audio) < expected_len:
        audio = np.pad(audio, (0, expected_len - len(audio)), mode="constant")

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
    )

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate time dimension to max_len
    if log_mel_spec.shape[1] < max_len:
        pad_width = max_len - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        log_mel_spec = log_mel_spec[:, :max_len]

    # Normalize to [0, 1]
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (
        log_mel_spec.max() - log_mel_spec.min() + 1e-8
    )

    # Add channel dimension for CNN: (n_mels, max_len, 1)
    return log_mel_spec[..., np.newaxis]


def augment_audio(audio, sr=SAMPLE_RATE):
    """
    Apply random augmentations to audio signal.

    Returns a list of augmented audio arrays (original + augmented variants).
    """
    augmented = []

    # Time stretch (faster/slower)
    for rate in [0.9, 1.1]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        augmented.append(stretched)

    # Pitch shift
    for n_steps in [-2, 2]:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        augmented.append(shifted)

    # Add gaussian noise
    noise = np.random.normal(0, 0.005, len(audio))
    augmented.append(audio + noise)

    return augmented


def prepare_dataset(metadata_csv=None, test_size=0.2, augment=True, random_state=42):
    """
    Process all audio files and create train/test splits.

    Args:
        metadata_csv: Path to metadata.csv. Defaults to backend/data/metadata.csv.
        test_size: Fraction of data for testing.
        augment: Whether to augment training data.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) shapes.
    """
    if metadata_csv is None:
        metadata_csv = os.path.join(BASE_DIR, "data", "metadata.csv")

    df = pd.read_csv(metadata_csv)
    print(f"Loading {len(df)} audio files...")

    features = []
    labels = []
    skipped = 0

    for _, row in df.iterrows():
        file_path = os.path.join(BASE_DIR, row["filepath"])

        if not os.path.exists(file_path):
            skipped += 1
            continue

        try:
            spec = load_and_extract_features(file_path)
            features.append(spec)
            labels.append(row["emotion_code"])
        except Exception as e:
            print(f"  Error processing {row['filename']}: {e}")
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} files")

    X = np.array(features)
    y = np.array(labels)

    print(f"Extracted features: {X.shape}, labels: {y.shape}")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Augment training data
    if augment:
        print("Augmenting training data...")
        X_aug, y_aug = _augment_training_data(X_train, y_train, df, metadata_csv)
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)
        print(f"After augmentation: {X_train.shape[0]} training samples")

    # One-hot encode labels
    from tensorflow.keras.utils import to_categorical

    num_classes = 4
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Save to disk
    train_dir = os.path.join(BASE_DIR, "data", "train")
    test_dir = os.path.join(BASE_DIR, "data", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.save(os.path.join(train_dir, "X_train.npy"), X_train)
    np.save(os.path.join(train_dir, "y_train.npy"), y_train_cat)
    np.save(os.path.join(test_dir, "X_test.npy"), X_test)
    np.save(os.path.join(test_dir, "y_test.npy"), y_test_cat)

    print(f"Saved: X_train{X_train.shape}, y_train{y_train_cat.shape}")
    print(f"Saved: X_test{X_test.shape}, y_test{y_test_cat.shape}")

    return X_train.shape, X_test.shape, y_train_cat.shape, y_test_cat.shape


def _augment_training_data(X_train, y_train, df, metadata_csv):
    """Generate augmented samples from training audio files."""
    X_aug_list = []
    y_aug_list = []

    # Re-read original audio for augmentation (spectrograms can't be reverse-engineered)
    metadata_csv_path = metadata_csv or os.path.join(BASE_DIR, "data", "metadata.csv")
    df = pd.read_csv(metadata_csv_path)

    # Get training file indices based on train labels count
    # We'll augment each training sample
    train_indices = list(range(len(y_train)))

    for idx in train_indices:
        # Map back to original file using the dataframe
        if idx >= len(df):
            continue

        row = df.iloc[idx]
        file_path = os.path.join(BASE_DIR, row["filepath"])

        if not os.path.exists(file_path):
            continue

        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            augmented_audios = augment_audio(audio, sr)

            for aug_audio in augmented_audios:
                # Pad to expected length
                expected_len = int(SAMPLE_RATE * DURATION)
                if len(aug_audio) < expected_len:
                    aug_audio = np.pad(
                        aug_audio, (0, expected_len - len(aug_audio)), mode="constant"
                    )
                elif len(aug_audio) > expected_len:
                    aug_audio = aug_audio[:expected_len]

                mel_spec = librosa.feature.melspectrogram(
                    y=aug_audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=2048, hop_length=512
                )
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)

                if log_mel.shape[1] < MAX_LEN:
                    log_mel = np.pad(
                        log_mel, ((0, 0), (0, MAX_LEN - log_mel.shape[1])), mode="constant"
                    )
                else:
                    log_mel = log_mel[:, :MAX_LEN]

                log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

                X_aug_list.append(log_mel[..., np.newaxis])
                y_aug_list.append(y_train[idx])

        except Exception as e:
            print(f"  Augmentation error for {row['filename']}: {e}")

    if X_aug_list:
        return np.array(X_aug_list), np.array(y_aug_list)
    return np.empty((0, N_MELS, MAX_LEN, 1)), np.empty(0, dtype=int)


def preprocess_single_file(file_path):
    """
    Preprocess a single WAV file for inference.

    Args:
        file_path: Path to the WAV file.

    Returns:
        np.ndarray of shape (1, n_mels, max_len, 1) — batch of 1 for prediction.
    """
    features = load_and_extract_features(file_path)
    return np.expand_dims(features, axis=0)


def preprocess_uploaded_files(file_paths, labels):
    """
    Preprocess a batch of uploaded WAV files for retraining.

    Args:
        file_paths: List of paths to WAV files.
        labels: List of integer labels (0=angry, 1=drunk, 2=painful, 3=stressful).

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    features = []
    valid_labels = []

    for fpath, label in zip(file_paths, labels):
        try:
            spec = load_and_extract_features(fpath)
            features.append(spec)
            valid_labels.append(label)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    if not features:
        return np.empty((0, N_MELS, MAX_LEN, 1)), np.empty(0, dtype=int)

    X = np.array(features)
    y = np.array(valid_labels)
    return X, y


if __name__ == "__main__":
    print("Running dataset preparation...")
    shapes = prepare_dataset()
    print(f"\nDataset prepared successfully!")
    print(f"Train: X={shapes[0]}, y={shapes[2]}")
    print(f"Test:  X={shapes[1]}, y={shapes[3]}")
