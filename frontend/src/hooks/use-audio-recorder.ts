"use client";

import { useState, useRef, useCallback } from "react";

interface AudioRecorderState {
  isRecording: boolean;
  audioBlob: Blob | null;
  audioUrl: string | null;
  duration: number;
}

export function useAudioRecorder(maxDuration: number = 3) {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    audioBlob: null,
    audioUrl: null,
    duration: 0,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });

      chunksRef.current = [];
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        const url = URL.createObjectURL(blob);

        setState((prev) => ({
          ...prev,
          isRecording: false,
          audioBlob: blob,
          audioUrl: url,
        }));

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      startTimeRef.current = Date.now();

      setState((prev) => ({
        ...prev,
        isRecording: true,
        audioBlob: null,
        audioUrl: null,
        duration: 0,
      }));

      // Update duration every 100ms
      timerRef.current = setInterval(() => {
        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        setState((prev) => ({ ...prev, duration: elapsed }));

        // Auto-stop after maxDuration
        if (elapsed >= maxDuration) {
          stopRecording();
        }
      }, 100);
    } catch (err) {
      console.error("Failed to start recording:", err);
    }
  }, [maxDuration]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    if (state.audioUrl) {
      URL.revokeObjectURL(state.audioUrl);
    }
    setState({
      isRecording: false,
      audioBlob: null,
      audioUrl: null,
      duration: 0,
    });
  }, [state.audioUrl]);

  return {
    ...state,
    startRecording,
    stopRecording,
    reset,
  };
}
