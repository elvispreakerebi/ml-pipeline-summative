"use client";

import { RetrainForm } from "@/components/retrain-form";

export default function RetrainPage() {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Retrain Model</h2>
        <p className="mt-2 text-muted-foreground">
          Upload new labeled audio data and trigger model retraining. The
          existing model is used as a pretrained base for fine-tuning.
        </p>
      </div>

      <RetrainForm />
    </div>
  );
}
