"use client";

import { useMutation } from "@tanstack/react-query";
import { retrainModel, type RetrainResult } from "@/lib/api";

export function useRetrain() {
  return useMutation<
    RetrainResult,
    Error,
    { files: File[]; labels: string[] }
  >({
    mutationFn: ({ files, labels }) => retrainModel(files, labels),
  });
}
