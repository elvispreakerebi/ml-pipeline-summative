"use client";

import { useCallback, useState } from "react";
import { motion } from "motion/react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

interface AudioDropzoneProps {
  onFileDrop: (file: File) => void;
  disabled?: boolean;
}

export function AudioDropzone({ onFileDrop, disabled }: AudioDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file && file.name.toLowerCase().endsWith(".wav")) {
        onFileDrop(file);
      }
    },
    [onFileDrop]
  );

  return (
    <motion.div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      animate={{
        borderColor: isDragOver ? "hsl(var(--primary))" : "hsl(var(--border))",
        scale: isDragOver ? 1.02 : 1,
      }}
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-8 transition-colors",
        isDragOver && "bg-accent/50",
        disabled && "pointer-events-none opacity-50"
      )}
    >
      <Upload className="h-8 w-8 text-muted-foreground" />
      <div className="text-center">
        <p className="text-sm font-medium">Drag & drop a WAV file here</p>
        <p className="text-xs text-muted-foreground">or use the buttons above</p>
      </div>
    </motion.div>
  );
}
