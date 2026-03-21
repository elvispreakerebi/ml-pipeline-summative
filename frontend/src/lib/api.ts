const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictionResult {
  class_label: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  spectrogram?: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  uptime_seconds: number;
  uptime_human: string;
  timestamp: string;
}

export interface Metrics {
  accuracy: number;
  loss: number;
  f1_score: number;
  precision: number;
  recall: number;
  confusion_matrix: number[][];
  evaluated_at: string;
}

export interface RetrainResult {
  status: string;
  files_processed: number;
  old_metrics: Partial<Metrics>;
  new_metrics: Partial<Metrics>;
  retrained_at: string;
}

export interface ClassDistribution {
  distribution: Record<string, number>;
  total: number;
}

export interface GenderDistribution {
  distribution: Record<string, Record<string, number>>;
}

export interface TypeDistribution {
  distribution: Record<string, number>;
}

export interface SampleSpectrograms {
  spectrograms: Record<string, string>;
}

export async function predictEmotion(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/predict/spectrogram`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Prediction failed" }));
    throw new Error(error.detail || "Prediction failed");
  }

  return res.json();
}

export async function retrainModel(
  files: File[],
  labels: string[]
): Promise<RetrainResult> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("labels", labels.join(","));

  const res = await fetch(`${API_URL}/retrain`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Retraining failed" }));
    throw new Error(error.detail || "Retraining failed");
  }

  return res.json();
}

export async function getHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_URL}/health`);
  return res.json();
}

export async function getMetrics(): Promise<Metrics> {
  const res = await fetch(`${API_URL}/metrics`);
  if (!res.ok) throw new Error("Metrics not available");
  return res.json();
}

export async function getClasses(): Promise<{ classes: string[] }> {
  const res = await fetch(`${API_URL}/classes`);
  return res.json();
}

export async function getClassDistribution(): Promise<ClassDistribution> {
  const res = await fetch(`${API_URL}/insights/class-distribution`);
  return res.json();
}

export async function getGenderDistribution(): Promise<GenderDistribution> {
  const res = await fetch(`${API_URL}/insights/gender-distribution`);
  return res.json();
}

export async function getTypeDistribution(): Promise<TypeDistribution> {
  const res = await fetch(`${API_URL}/insights/type-distribution`);
  return res.json();
}

export async function getSampleSpectrograms(): Promise<SampleSpectrograms> {
  const res = await fetch(`${API_URL}/insights/sample-spectrograms`);
  return res.json();
}
