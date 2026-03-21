"use client";

import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MetricsCards } from "@/components/metrics-cards";
import { ConfusionMatrix } from "@/components/confusion-matrix";
import { useMetrics, useHealth } from "@/hooks/use-metrics";

export default function StatusPage() {
  const metrics = useMetrics();
  const health = useHealth();

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Model Status</h2>
        <p className="mt-2 text-muted-foreground">
          Monitor model health, uptime, and evaluation metrics.
        </p>
      </div>

      {/* Health */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Service Health</CardTitle>
          </CardHeader>
          <CardContent>
            {health.isLoading ? (
              <div className="h-20 animate-pulse rounded bg-muted" />
            ) : health.data ? (
              <div className="flex flex-wrap items-center gap-6">
                <div className="flex items-center gap-2">
                  <div
                    className={`h-3 w-3 rounded-full ${
                      health.data.model_loaded ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span className="text-sm">
                    {health.data.model_loaded ? "Model Loaded" : "No Model"}
                  </span>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Uptime: </span>
                  <Badge variant="secondary">{health.data.uptime_human}</Badge>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Status: </span>
                  <Badge
                    variant={
                      health.data.status === "ok" ? "default" : "destructive"
                    }
                  >
                    {health.data.status}
                  </Badge>
                </div>
              </div>
            ) : (
              <p className="text-sm text-destructive">
                API unavailable. Make sure the backend is running.
              </p>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Metrics */}
      {metrics.data && (
        <>
          <MetricsCards
            accuracy={metrics.data.accuracy}
            f1Score={metrics.data.f1_score}
            precision={metrics.data.precision}
            recall={metrics.data.recall}
          />

          {/* Confusion Matrix */}
          {metrics.data.confusion_matrix && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Confusion Matrix</CardTitle>
                </CardHeader>
                <CardContent>
                  <ConfusionMatrix matrix={metrics.data.confusion_matrix} />
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Loss */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Model Loss
                    </p>
                    <p className="mt-1 text-2xl font-bold">
                      {metrics.data.loss.toFixed(4)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-muted-foreground">
                      Last Evaluated
                    </p>
                    <p className="mt-1 text-sm">
                      {new Date(metrics.data.evaluated_at).toLocaleString()}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}

      {metrics.isError && (
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground">
            No metrics available yet. Train and evaluate the model first.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
