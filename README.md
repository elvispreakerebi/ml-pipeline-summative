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

Option A — Kaggle CLI:
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
python scripts/download_data.py
```

Option B — Manual:
1. Download from https://www.kaggle.com/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls
2. Extract to `backend/data/raw/CUSTOM_DATASET/`
3. Run `python scripts/download_data.py --skip-download`

### 4. Train the model

```bash
cd backend
python -m src.preprocessing   # Generate spectrograms
python -m src.model            # Train CNN
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

## Load Testing Results

TODO: Add Locust results with different container counts

## Model Evaluation

| Metric | Score |
|--------|-------|
| Accuracy | TODO |
| F1 Score | TODO |
| Precision | TODO |
| Recall | TODO |
| Loss | TODO |
