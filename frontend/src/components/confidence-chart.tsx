"use client";

import { motion } from "motion/react";

interface ConfidenceChartProps {
  probabilities: Record<string, number>;
}

const EMOTION_COLORS: Record<string, string> = {
  angry: "#EF4444",
  drunk: "#F59E0B",
  painful: "#8B5CF6",
  stressful: "#3B82F6",
};

export function ConfidenceChart({ probabilities }: ConfidenceChartProps) {
  const sorted = Object.entries(probabilities).sort(([, a], [, b]) => b - a);

  return (
    <div className="space-y-3">
      {sorted.map(([emotion, confidence], i) => (
        <div key={emotion} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="font-medium capitalize">{emotion}</span>
            <span className="text-muted-foreground">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidence * 100}%` }}
              transition={{ duration: 0.6, delay: i * 0.1, ease: "easeOut" }}
              className="h-full rounded-full"
              style={{ backgroundColor: EMOTION_COLORS[emotion] || "#6B7280" }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
