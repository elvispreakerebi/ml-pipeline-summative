# Emergency Call Speech Emotion Recognition

An end-to-end ML pipeline that classifies emotions in emergency call audio recordings. The system detects 4 emotional states — **angry**, **drunk**, **painful**, and **stressful** — to help emergency dispatch systems prioritize responses.

## Demo

- **Video Demo:** [YouTube Link](TODO)
- **Live App:** [Render URL](TODO)

## Dataset

[Speech Emotion Recognition for Emergency Calls](https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls) — ~326 WAV files from 18 speakers, 4 emotion classes, 4 emergency sentences (~3 sec each).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | TensorFlow/Keras |
| Audio Processing | librosa, audiomentations |
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js, shadcn/ui, Tailwind CSS, TanStack Query, Motion |
| Containerization | Docker + Docker Compose |
| Load Testing | Locust |
| Deployment | Render |

## Project Structure

```
ml-pipeline-summative/
├── backend/
│   ├── notebook/ml_pipeline.ipynb    # Full ML pipeline documentation
│   ├── src/
│   │   ├── preprocessing.py          # Audio → mel spectrogram
│   │   ├── model.py                  # CNN architecture + training
│   │   └── prediction.py             # Inference logic
│   ├── api/main.py                   # FastAPI endpoints
│   ├── data/                         # Train/test data
│   ├── models/                       # Saved .h5 model
│   ├── locust/                       # Load testing
│   └── Dockerfile
├── frontend/                         # Next.js dashboard
│   ├── src/app/                      # Pages (predict, retrain, insights, status)
│   └── Dockerfile
├── docker-compose.yml
└── render.yaml
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 20+
- Docker (optional)

### 1. Clone and install

```bash
git clone <repo-url>
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

1. Download from https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls
2. Extract `CUSTOM_DATASET/` folder into `backend/data/raw/`
3. Generate metadata:
```bash
cd backend
python scripts/download_data.py
```

### 4. Train the model

```bash
cd backend
python -m src.preprocessing   # Generate mel spectrograms + augmentation
python -m src.experiments      # Run 5 experiments, select best model
```

### 5. Run the API

```bash
cd backend
uvicorn api.main:app --reload --port 8000
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
docker-compose up --build
```
- API: http://localhost:8000
- Frontend: http://localhost:3000

## Features

- **Predict Emotion** — Upload a WAV file, record audio, or drag-and-drop to classify the caller's emotion
- **Retrain Model** — Upload new labeled audio data and trigger model retraining
- **Data Insights** — Visualizations of class distribution, gender breakdown, spectrograms
- **Model Status** — Live accuracy, F1, precision, recall metrics and confusion matrix

## Model Evaluation

Best model: **exp2_deeper_l2** (Deeper CNN + L2 regularization), selected from 5 experiments by F1 score.

| Metric | Score |
|--------|-------|
| Accuracy | 42.65% |
| F1 Score | 0.3666 |
| Precision | 0.3571 |
| Recall | 0.4216 |
| Loss | 1.7611 |

### Per-Class Performance

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Angry | 0.43 | 0.83 | 0.57 |
| Drunk | 0.73 | 0.50 | 0.59 |
| Painful | 0.27 | 0.35 | 0.31 |
| Stressful | 0.00 | 0.00 | 0.00 |

### Experiment Comparison

| Experiment | Accuracy | F1 | Params |
|------------|----------|-----|--------|
| exp2_deeper_l2 | 0.4265 | 0.3666 | 489,988 |
| exp5_separable_conv | 0.3529 | 0.3350 | 37,540 |
| exp1_baseline_cnn | 0.3382 | 0.3193 | 110,596 |
| exp3_small_aggressive_dropout | 0.3382 | 0.3168 | 28,164 |
| exp4_large_kernel_sgd | 0.2500 | 0.2327 | 144,388 |

> **Note:** Accuracy is modest (~43%) due to the very small dataset (338 samples, 4 classes). The model performs best on "angry" and "drunk" emotions. Performance can be improved with more training data via the retraining pipeline.
