"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Upload, X, Loader2, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useRetrain } from "@/hooks/use-retrain";

const EMOTIONS = ["angry", "drunk", "painful", "stressful"];

interface FileEntry {
  file: File;
  label: string;
  id: string;
}

export function RetrainForm() {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const retrain = useRetrain();

  const handleFileAdd = useCallback((newFiles: FileList | null) => {
    if (!newFiles) return;

    const entries: FileEntry[] = Array.from(newFiles)
      .filter((f) => f.name.toLowerCase().endsWith(".wav"))
      .map((f) => ({
        file: f,
        label: "angry",
        id: `${f.name}-${Date.now()}-${Math.random()}`,
      }));

    setFiles((prev) => [...prev, ...entries]);
  }, []);

  const handleLabelChange = useCallback((id: string, label: string) => {
    setFiles((prev) =>
      prev.map((f) => (f.id === id ? { ...f, label } : f))
    );
  }, []);

  const handleRemove = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const handleRetrain = useCallback(() => {
    retrain.mutate({
      files: files.map((f) => f.file),
      labels: files.map((f) => f.label),
    });
  }, [files, retrain]);

  return (
    <div className="space-y-6">
      {/* Upload area */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Upload Training Data</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 transition-colors hover:border-primary hover:bg-accent/30"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFileAdd(e.dataTransfer.files);
            }}
          >
            <Upload className="h-8 w-8 text-muted-foreground" />
            <p className="text-sm font-medium">
              Drop WAV files here or click to browse
            </p>
            <p className="text-xs text-muted-foreground">
              Upload multiple audio files with their emotion labels
            </p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav"
            multiple
            onChange={(e) => handleFileAdd(e.target.files)}
            className="hidden"
          />
        </CardContent>
      </Card>

      {/* File list */}
      {files.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">
              Files ({files.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <AnimatePresence>
                {files.map((entry) => (
                  <motion.div
                    key={entry.id}
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex items-center gap-3 rounded-lg border p-3"
                  >
                    <div className="flex-1 text-sm">
                      <p className="font-medium">{entry.file.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(entry.file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>

                    <select
                      value={entry.label}
                      onChange={(e) =>
                        handleLabelChange(entry.id, e.target.value)
                      }
                      className="rounded-md border bg-background px-3 py-1.5 text-sm"
                    >
                      {EMOTIONS.map((e) => (
                        <option key={e} value={e}>
                          {e}
                        </option>
                      ))}
                    </select>

                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleRemove(entry.id)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

            <div className="mt-6">
              <Button
                onClick={handleRetrain}
                disabled={retrain.isPending || files.length === 0}
                size="lg"
                className="gap-2"
              >
                {retrain.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Retraining...
                  </>
                ) : (
                  "Start Retraining"
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {retrain.data && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Retraining Complete
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4 text-sm text-muted-foreground">
                Processed {retrain.data.files_processed} files. Model updated.
              </p>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border p-4">
                  <p className="text-xs font-medium text-muted-foreground">
                    Before
                  </p>
                  <div className="mt-2 space-y-1">
                    <MetricRow
                      label="Accuracy"
                      value={retrain.data.old_metrics.accuracy}
                    />
                    <MetricRow
                      label="F1 Score"
                      value={retrain.data.old_metrics.f1_score}
                    />
                  </div>
                </div>
                <div className="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-900 dark:bg-green-950">
                  <p className="text-xs font-medium text-muted-foreground">
                    After
                  </p>
                  <div className="mt-2 space-y-1">
                    <MetricRow
                      label="Accuracy"
                      value={retrain.data.new_metrics.accuracy}
                    />
                    <MetricRow
                      label="F1 Score"
                      value={retrain.data.new_metrics.f1_score}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {retrain.isError && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
          {retrain.error.message}
        </div>
      )}
    </div>
  );
}

function MetricRow({
  label,
  value,
}: {
  label: string;
  value: number | undefined | null;
}) {
  return (
    <div className="flex justify-between text-sm">
      <span>{label}</span>
      <span className="font-mono font-medium">
        {value != null ? `${(value * 100).toFixed(1)}%` : "N/A"}
      </span>
    </div>
  );
}
