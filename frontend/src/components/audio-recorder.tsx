"use client";

import { motion } from "motion/react";
import { Mic, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAudioRecorder } from "@/hooks/use-audio-recorder";

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  disabled?: boolean;
}

export function AudioRecorder({ onRecordingComplete, disabled }: AudioRecorderProps) {
  const { isRecording, audioBlob, duration, startRecording, stopRecording } =
    useAudioRecorder(3);

  // When recording completes, notify parent
  if (audioBlob) {
    onRecordingComplete(audioBlob);
  }

  return (
    <div className="flex items-center gap-4">
      {!isRecording ? (
        <Button
          onClick={startRecording}
          disabled={disabled}
          variant="outline"
          size="lg"
          className="gap-2"
        >
          <Mic className="h-4 w-4" />
          Record Audio
        </Button>
      ) : (
        <div className="flex items-center gap-3">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 1 }}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500"
          >
            <Mic className="h-5 w-5 text-white" />
          </motion.div>

          <div className="text-sm">
            <span className="font-medium text-red-500">Recording...</span>
            <span className="ml-2 text-muted-foreground">
              {duration.toFixed(1)}s / 3.0s
            </span>
          </div>

          <Button onClick={stopRecording} variant="destructive" size="sm">
            <Square className="mr-1 h-3 w-3" />
            Stop
          </Button>
        </div>
      )}
    </div>
  );
}
