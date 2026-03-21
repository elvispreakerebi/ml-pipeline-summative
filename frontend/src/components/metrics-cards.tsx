"use client";

import { motion } from "motion/react";
import { Card, CardContent } from "@/components/ui/card";

interface MetricCardProps {
  label: string;
  value: number;
  delay?: number;
}

function MetricCard({ label, value, delay = 0 }: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
    >
      <Card>
        <CardContent className="p-6">
          <p className="text-sm font-medium text-muted-foreground">{label}</p>
          <motion.p
            className="mt-2 text-3xl font-bold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: delay + 0.2 }}
          >
            {(value * 100).toFixed(1)}%
          </motion.p>
        </CardContent>
      </Card>
    </motion.div>
  );
}

interface MetricsCardsProps {
  accuracy: number;
  f1Score: number;
  precision: number;
  recall: number;
}

export function MetricsCards({
  accuracy,
  f1Score,
  precision,
  recall,
}: MetricsCardsProps) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard label="Accuracy" value={accuracy} delay={0} />
      <MetricCard label="F1 Score" value={f1Score} delay={0.1} />
      <MetricCard label="Precision" value={precision} delay={0.2} />
      <MetricCard label="Recall" value={recall} delay={0.3} />
    </div>
  );
}
