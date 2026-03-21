# Emergency Call Speech Emotion Recognition

An end-to-end ML pipeline that classifies emotions in emergency call audio recordings. The system detects 4 emotional states — **angry**, **drunk**, **painful**, and **stressful** — to help emergency dispatch systems prioritize responses.

## Demo

- **Video Demo:** [YouTube Link](TODO)
- **Live API:** [https://emotion-api.onrender.com](https://emotion-api.onrender.com)
- **Live Frontend:** [https://emotion-frontend.onrender.com](https://emotion-frontend.onrender.com)

---

## Problem Statement

Emergency dispatch systems receive thousands of calls daily. A caller screaming in **pain** or under extreme **stress** may need a faster response than one who is simply **angry**. By automatically detecting the caller's emotional state from their voice, dispatchers can triage calls more effectively and potentially save lives.

This project builds a complete ML pipeline — from raw audio to a deployed web application — that classifies 3-second emergency call audio clips into one of 4 emotion categories.

---

## Dataset

**[Speech Emotion Recognition for Emergency Calls](https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls)**

| Property | Value |
|----------|-------|
| Total samples | ~326 WAV files |
| Speakers | 18 (male, female, unknown) |
| Emotion classes | 4 (angry, drunk, painful, stressful) |
| Sentences | 4 emergency phrases |
| Duration | ~3 seconds per clip |
| Recording types | Natural + Synthetic (pitch-shifted) |
| Size | ~146 MB |

### Filename Convention
`EmotionNum_SentenceNum_Gender_SyntheticNatural_SpeakerNum.wav`
- Emotion: 01=angry, 02=drunk, 03=painful, 04=stressful
- Gender: 01=female, 02=male
- Type: 01=natural, 02=synthetic

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | TensorFlow 2.20 / Keras |
| Audio Processing | librosa, audiomentations |
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js 14, shadcn/ui, Tailwind CSS, TanStack Query, Motion |
| Containerization | Docker + Docker Compose |
| Load Testing | Locust |
| Deployment | Render |

---

## Project Structure

```
ml-pipeline-summative/
├── backend/
│   ├── notebook/
│   │   └── ml_pipeline.ipynb        # Full ML pipeline (EDA → training → evaluation)
│   ├── src/
│   │   ├── preprocessing.py         # WAV → mel spectrogram + augmentation
│   │   ├── model.py                 # CNN architecture, training, evaluation, retraining
│   │   ├── prediction.py            # Inference logic + class mapping
│   │   ├── experiments.py           # 5-experiment comparison pipeline
│   │   ├── tuning.py                # Hyperparameter grid search
│   │   └── visualization.py         # Plots (confusion matrix, spectrograms, etc.)
│   ├── api/
│   │   └── main.py                  # FastAPI endpoints (predict, retrain, metrics, insights)
│   ├── data/
│   │   ├── raw/CUSTOM_DATASET/      # Original WAV files (18 speaker folders)
│   │   ├── train/                   # X_train.npy, y_train.npy
│   │   ├── test/                    # X_test.npy, y_test.npy
│   │   └── metadata.csv            # Parsed file metadata
│   ├── models/
│   │   ├── emotion_classifier.h5    # Best model (deployed)
│   │   ├── exp[1-5]_*.h5           # All 5 experiment models
│   │   ├── experiment_results.json  # Experiment comparison data
│   │   ├── metrics.json             # Current evaluation metrics
│   │   └── tuning_results.json      # Hyperparameter search results
│   ├── scripts/
│   │   └── download_data.py         # Dataset download + metadata generation
│   ├── locust/
│   │   └── locustfile.py            # Load testing configuration
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/                     # Next.js App Router pages
│   │   │   ├── page.tsx             # Prediction page (home)
│   │   │   ├── retrain/page.tsx     # Model retraining
│   │   │   ├── insights/page.tsx    # Data visualizations
│   │   │   └── status/page.tsx      # Model status & metrics
│   │   ├── components/              # UI components
│   │   │   ├── audio-recorder.tsx   # Live microphone recording
│   │   │   ├── audio-dropzone.tsx   # Drag-and-drop WAV upload
│   │   │   ├── prediction-result.tsx
│   │   │   ├── confidence-chart.tsx
│   │   │   ├── spectrogram-viewer.tsx
│   │   │   ├── retrain-form.tsx
│   │   │   ├── metrics-cards.tsx
│   │   │   ├── confusion-matrix.tsx
│   │   │   └── nav-sidebar.tsx
│   │   ├── hooks/                   # TanStack Query hooks
│   │   └── lib/api.ts               # API client
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── render.yaml
```

---

## Features

### Prediction
Upload a WAV file, record live audio from your microphone, or drag-and-drop to classify the caller's emotional state. Results include:
- Predicted emotion with confidence percentage
- Probability distribution across all 4 classes (bar chart)
- Mel spectrogram visualization of the uploaded audio

### Retraining
Upload new labeled WAV files to fine-tune the model:
1. Upload WAV files and assign emotion labels
2. Model loads pretrained weights, freezes early layers
3. Fine-tunes on new data combined with 30% of original training data
4. Displays before/after metrics comparison

### Data Insights
Interactive visualizations of the dataset:
- Emotion class distribution
- Gender distribution per emotion
- Natural vs synthetic recording breakdown
- Sample spectrograms per emotion class

### Model Status
Live model performance dashboard:
- Accuracy, F1 Score, Precision, Recall cards
- Confusion matrix heatmap
- Per-class classification report
- Model uptime and health status

---

## ML Pipeline

### 1. Preprocessing
- Load WAV audio at 22,050 Hz, clip/pad to 3 seconds
- Compute 128-band mel spectrogram
- Convert to log scale (dB)
- Pad/truncate to 128 time frames → 128×128×1 grayscale image
- Normalize to [0, 1]

### 2. Data Augmentation (5x expansion)
Critical for the small dataset (~326 → ~1,620 training samples):
- Time stretching (0.9x, 1.1x speed)
- Pitch shifting (±2 semitones)
- Gaussian noise injection

### 3. Model Architecture
CNN with optimization techniques:
- 3 Conv2D blocks with BatchNormalization + MaxPooling + Dropout (0.25)
- GlobalAveragePooling2D (fewer parameters than Flatten)
- Dense layers with Dropout (0.5) and L2 regularization
- Softmax output for 4 classes

### 4. Training — 5 Experiments
Each experiment uses a different architecture/optimizer combination:

| Experiment | Description | Accuracy | F1 Score | Params |
|-----------|-------------|----------|----------|--------|
| **exp4_large_kernel_sgd** | **5×5 kernels + SGD momentum** | **36.8%** | **0.315** | **144K** |
| exp2_deeper_l2 | 4 conv blocks + L2 reg, Adam | 30.9% | 0.296 | 490K |
| exp5_separable_conv | SeparableConv2D, Adam | 29.4% | 0.288 | 38K |
| exp1_baseline_cnn | 3 conv blocks, Adam | 33.8% | 0.274 | 111K |
| exp3_small_aggressive_dropout | Small CNN + heavy dropout | 30.9% | 0.228 | 28K |

Best model selected by **macro F1 score**: `exp4_large_kernel_sgd`

### 5. Hyperparameter Tuning
Grid search over 8 combinations:
- Learning rate: [0.001, 0.0001]
- Batch size: [8, 16]
- Dropout rate: [0.25, 0.5]

**Best config:** LR=0.001, batch=8, dropout=0.5 (38.2% accuracy)

### 6. Evaluation

| Metric | Score |
|--------|-------|
| **Accuracy** | 36.8% |
| **F1 Score** (macro) | 0.315 |
| **Precision** (macro) | 0.337 |
| **Recall** (macro) | 0.356 |
| **Loss** | 1.362 |

#### Per-Class Performance

| Emotion | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Angry | 0.39 | 0.83 | 0.54 |
| Drunk | 0.13 | 0.06 | 0.08 |
| Painful | 0.44 | 0.24 | 0.31 |
| Stressful | 0.38 | 0.29 | 0.33 |

> **Note:** ~37% accuracy on 4 classes (random chance = 25%) with only ~326 samples is expected. The main bottleneck is dataset size — 18 speakers with 4 sentences provide limited vocal variation. The model correctly identifies **angry** emotion 83% of the time due to its distinct high-energy spectral patterns.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Upload WAV → emotion prediction + probabilities |
| `POST` | `/predict/spectrogram` | Upload WAV → base64 mel spectrogram image |
| `POST` | `/retrain` | Upload labeled WAVs → retrain model → new metrics |
| `GET` | `/health` | Server status, model loaded, uptime |
| `GET` | `/metrics` | Current model evaluation metrics |
| `GET` | `/classes` | List of 4 emotion classes |
| `GET` | `/insights/class-distribution` | Emotion class counts |
| `GET` | `/insights/gender-distribution` | Gender breakdown per emotion |
| `GET` | `/insights/sample-spectrograms` | Base64 spectrograms (one per class) |

---

## Setup

### Prerequisites
- Python 3.10+ (tested with 3.13)
- Node.js 20+
- Docker (optional, for containerized deployment)

### 1. Clone the repository

```bash
git clone https://github.com/elvispreakerebi/ml-pipeline-summative.git
cd ml-pipeline-summative
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls)
2. Extract `CUSTOM_DATASET/` folder into `backend/data/raw/`
3. Generate metadata:
```bash
python scripts/download_data.py
```

### 4. Train the model

```bash
# Preprocess audio → mel spectrograms (with augmentation)
python -c "from src.preprocessing import prepare_dataset; prepare_dataset('data/metadata.csv')"

# Run 5 experiments and save best model
python -m src.experiments
```

Or run the full pipeline interactively via the Jupyter notebook:
```bash
jupyter notebook notebook/ml_pipeline.ipynb
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```
API available at http://localhost:8000 — test with:
```bash
curl http://localhost:8000/health
curl -X POST -F "file=@path/to/audio.wav" http://localhost:8000/predict
```

### 6. Frontend setup

```bash
cd frontend
npm install
npm run dev
```
Visit http://localhost:3000

### 7. Docker (alternative)

```bash
# From project root
docker-compose up --build
```
- API: http://localhost:8000
- Frontend: http://localhost:3000

---

## Deployment

Deployed on **Render** using `render.yaml`:
- **Backend (emotion-api):** Docker web service running FastAPI on port 8000
- **Frontend (emotion-frontend):** Docker web service running Next.js on port 3000

Environment variables:
- `MODEL_PATH`: Path to .h5 model inside the container
- `NEXT_PUBLIC_API_URL`: Backend API URL for the frontend to connect to

> **Note:** On Render's free tier, retraining works in-memory but resets on container restart (no persistent storage).

---

## Load Testing

Using Locust to benchmark API performance:

```bash
cd backend
locust -f locust/locustfile.py --host=http://localhost:8000
```

Open http://localhost:8089 to configure and run load tests. Tests include health checks, predictions with sample WAV files, and metrics retrieval.

---

## Notebook

The Jupyter notebook (`backend/notebook/ml_pipeline.ipynb`) documents the complete pipeline:

1. **Setup & Imports** — Environment configuration
2. **Data Loading** — Metadata parsing from filenames
3. **EDA** — Class distribution, gender breakdown, waveforms, spectrograms
4. **Preprocessing** — Mel spectrogram extraction + augmentation demo
5. **Model Architecture** — CNN design with optimization techniques
6. **Training** — 5 experiments with different architectures
7. **Evaluation** — 5 metrics + confusion matrix + per-class report
8. **Hyperparameter Tuning** — Grid search (8 combinations)
9. **Prediction Demo** — Inference on test samples
10. **Retraining Demo** — Fine-tuning with new data
11. **Conclusions** — Key findings + improvement suggestions

---

## Author

Elvis Preaker Ebi — ALU BSE Program, Machine Learning Pipelines Course
