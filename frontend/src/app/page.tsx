"use client";

import { useState, useRef, useCallback } from "react";
import { motion } from "motion/react";
import { Upload, Mic, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { AudioDropzone } from "@/components/audio-dropzone";
import { AudioRecorder } from "@/components/audio-recorder";
import { PredictionResultDisplay } from "@/components/prediction-result";
import { usePrediction } from "@/hooks/use-prediction";

export default function PredictPage() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const prediction = usePrediction();

  const handleFileSelect = useCallback((file: File) => {
    setAudioFile(file);
    setAudioUrl(URL.createObjectURL(file));
    prediction.reset();
  }, [prediction]);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleRecordingComplete = useCallback(
    (blob: Blob) => {
      const file = new File([blob], "recording.wav", { type: "audio/wav" });
      handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleAnalyze = useCallback(() => {
    if (audioFile) {
      prediction.mutate(audioFile);
    }
  }, [audioFile, prediction]);

  const handleReset = useCallback(() => {
    setAudioFile(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    prediction.reset();
  }, [audioUrl, prediction]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Predict Emotion</h2>
        <p className="mt-2 text-muted-foreground">
          Upload, record, or drag-and-drop an audio file to classify the
          caller&apos;s emotion.
        </p>
      </div>

      {/* Audio Input Section */}
      <Card>
        <CardContent className="space-y-6 p-6">
          {/* Upload + Record buttons */}
          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              size="lg"
              className="gap-2"
            >
              <Upload className="h-4 w-4" />
              Upload WAV
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav"
              onChange={handleFileInput}
              className="hidden"
            />

            <AudioRecorder
              onRecordingComplete={handleRecordingComplete}
              disabled={prediction.isPending}
            />
          </div>

          {/* Drag & Drop zone */}
          <AudioDropzone
            onFileDrop={handleFileSelect}
            disabled={prediction.isPending}
          />

          {/* Audio Player + Analyze */}
          {audioFile && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div className="flex items-center gap-4 rounded-lg border bg-secondary/30 p-4">
                <div className="flex-1">
                  <p className="text-sm font-medium">{audioFile.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(audioFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                {audioUrl && (
                  <audio controls src={audioUrl} className="h-10" />
                )}
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={prediction.isPending}
                  size="lg"
                  className="gap-2"
                >
                  {prediction.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Mic className="h-4 w-4" />
                      Analyze Emotion
                    </>
                  )}
                </Button>
                <Button onClick={handleReset} variant="ghost" size="lg">
                  Clear
                </Button>
              </div>
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Error */}
      {prediction.isError && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive"
        >
          {prediction.error.message}
        </motion.div>
      )}

      {/* Results */}
      {prediction.data && <PredictionResultDisplay result={prediction.data} />}
    </div>
  );
}
