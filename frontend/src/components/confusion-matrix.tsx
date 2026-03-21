"use client";

import { cn } from "@/lib/utils";

const LABELS = ["angry", "drunk", "painful", "stressful"];

interface ConfusionMatrixProps {
  matrix: number[][];
}

export function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="p-2 text-xs text-muted-foreground"></th>
            {LABELS.map((label) => (
              <th
                key={label}
                className="p-2 text-center text-xs font-medium capitalize text-muted-foreground"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-2 text-right text-xs font-medium capitalize text-muted-foreground">
                {LABELS[i]}
              </td>
              {row.map((val, j) => {
                const intensity = maxVal > 0 ? val / maxVal : 0;
                return (
                  <td key={j} className="p-1">
                    <div
                      className={cn(
                        "flex h-12 w-full items-center justify-center rounded text-sm font-medium",
                        i === j ? "text-white" : "text-foreground"
                      )}
                      style={{
                        backgroundColor:
                          i === j
                            ? `rgba(59, 130, 246, ${0.2 + intensity * 0.8})`
                            : `rgba(239, 68, 68, ${intensity * 0.5})`,
                      }}
                    >
                      {val}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex justify-between text-xs text-muted-foreground">
        <span>Rows: Actual</span>
        <span>Columns: Predicted</span>
      </div>
    </div>
  );
}
