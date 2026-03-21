"use client";

import { useMutation } from "@tanstack/react-query";
import { predictEmotion, type PredictionResult } from "@/lib/api";

export function usePrediction() {
  return useMutation<PredictionResult, Error, File>({
    mutationFn: predictEmotion,
  });
}
