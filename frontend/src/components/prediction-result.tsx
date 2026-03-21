"use client";

import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ConfidenceChart } from "@/components/confidence-chart";
import { SpectrogramViewer } from "@/components/spectrogram-viewer";
import type { PredictionResult } from "@/lib/api";

const EMOTION_EMOJIS: Record<string, string> = {
  angry: "\uD83D\uDE21",
  drunk: "\uD83C\uDF7A",
  painful: "\uD83D\uDE23",
  stressful: "\uD83D\uDE30",
};

interface PredictionResultDisplayProps {
  result: PredictionResult;
}

export function PredictionResultDisplay({ result }: PredictionResultDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="grid gap-6 md:grid-cols-2"
    >
      {/* Prediction */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Predicted Emotion</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <span className="text-5xl">
                {EMOTION_EMOJIS[result.class_label] || "\uD83C\uDFA4"}
              </span>
              <div>
                <p className="text-2xl font-bold capitalize">
                  {result.class_label}
                </p>
                <Badge variant="secondary" className="mt-1">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Confidence Chart */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Confidence Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ConfidenceChart probabilities={result.probabilities} />
          </CardContent>
        </Card>
      </motion.div>

      {/* Spectrogram */}
      {result.spectrogram && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="md:col-span-2"
        >
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Mel Spectrogram</CardTitle>
            </CardHeader>
            <CardContent>
              <SpectrogramViewer base64Image={result.spectrogram} />
            </CardContent>
          </Card>
        </motion.div>
      )}
    </motion.div>
  );
}
