"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "motion/react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

interface AudioDropzoneProps {
  onFileDrop: (file: File) => void;
  disabled?: boolean;
}

export function AudioDropzone({ onFileDrop, disabled }: AudioDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        onFileDrop(file);
      }
    },
    [onFileDrop]
  );

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onFileDrop(file);
      }
      // Reset so the same file can be re-selected
      e.target.value = "";
    },
    [onFileDrop]
  );

  return (
    <motion.div
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      animate={{
        borderColor: isDragOver ? "hsl(var(--primary))" : "hsl(var(--border))",
        scale: isDragOver ? 1.02 : 1,
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-8 transition-colors hover:bg-accent/30",
        isDragOver && "bg-accent/50",
        disabled && "pointer-events-none opacity-50"
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".wav"
        onChange={handleFileInput}
        className="hidden"
      />
      <Upload className="h-8 w-8 text-muted-foreground" />
      <div className="text-center">
        <p className="text-sm font-medium">Drag & drop a WAV file here</p>
        <p className="text-xs text-muted-foreground">or click to browse</p>
      </div>
    </motion.div>
  );
}
