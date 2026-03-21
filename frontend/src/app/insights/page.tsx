"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { SpectrogramViewer } from "@/components/spectrogram-viewer";
import {
  getClassDistribution,
  getGenderDistribution,
  getTypeDistribution,
  getSampleSpectrograms,
} from "@/lib/api";

const EMOTION_COLORS: Record<string, string> = {
  angry: "#EF4444",
  drunk: "#F59E0B",
  painful: "#8B5CF6",
  stressful: "#3B82F6",
};

const PIE_COLORS = ["#10B981", "#6366F1"];

export default function InsightsPage() {
  const classDist = useQuery({
    queryKey: ["class-distribution"],
    queryFn: getClassDistribution,
  });

  const genderDist = useQuery({
    queryKey: ["gender-distribution"],
    queryFn: getGenderDistribution,
  });

  const typeDist = useQuery({
    queryKey: ["type-distribution"],
    queryFn: getTypeDistribution,
  });

  const spectrograms = useQuery({
    queryKey: ["sample-spectrograms"],
    queryFn: getSampleSpectrograms,
  });

  // Transform class distribution for Recharts
  const classData = classDist.data
    ? Object.entries(classDist.data.distribution).map(([emotion, count]) => ({
        emotion,
        count,
        fill: EMOTION_COLORS[emotion] || "#6B7280",
      }))
    : [];

  // Transform gender distribution for Recharts
  const genderData = genderDist.data
    ? Object.entries(genderDist.data.distribution.female || {}).map(
        ([emotion]) => ({
          emotion,
          female: genderDist.data!.distribution.female?.[emotion] || 0,
          male: genderDist.data!.distribution.male?.[emotion] || 0,
        })
      )
    : [];

  // Transform type distribution for PieChart
  const typeData = typeDist.data
    ? Object.entries(typeDist.data.distribution).map(([name, value]) => ({
        name,
        value,
      }))
    : [];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Data Insights</h2>
        <p className="mt-2 text-muted-foreground">
          Explore the emergency call emotion dataset through visualizations.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Visualization 1: Class Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Emotion Class Distribution</CardTitle>
              <CardDescription>
                Number of audio samples per emotion category. A balanced
                distribution helps the model learn each class equally well.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {classDist.isLoading ? (
                <div className="h-64 animate-pulse rounded bg-muted" />
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={classData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="emotion" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {classData.map((entry) => (
                        <Cell key={entry.emotion} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Visualization 2: Gender Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Gender Distribution per Emotion</CardTitle>
              <CardDescription>
                Breakdown of male and female speakers for each emotion.
                Gender balance ensures the model doesn&apos;t develop gender bias.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {genderDist.isLoading ? (
                <div className="h-64 animate-pulse rounded bg-muted" />
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={genderData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="emotion" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="female" fill="#EC4899" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="male" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Visualization 3: Natural vs Synthetic */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Natural vs Synthetic Audio</CardTitle>
              <CardDescription>
                Synthetic audio was created by pitch-shifting natural recordings.
                This augmentation helps increase dataset size and model robustness.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {typeDist.isLoading ? (
                <div className="h-64 animate-pulse rounded bg-muted" />
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={typeData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      dataKey="value"
                      label={({ name, percent }) =>
                        `${name} (${(percent * 100).toFixed(0)}%)`
                      }
                    >
                      {typeData.map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Visualization 4: Sample Spectrograms */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Sample Spectrograms</CardTitle>
              <CardDescription>
                Mel spectrograms show how audio frequency content varies over
                time. Each emotion has distinct visual patterns the CNN learns
                to recognize.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {spectrograms.isLoading ? (
                <div className="grid grid-cols-2 gap-2">
                  {[0, 1, 2, 3].map((i) => (
                    <div key={i} className="h-32 animate-pulse rounded bg-muted" />
                  ))}
                </div>
              ) : spectrograms.data ? (
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(spectrograms.data.spectrograms).map(
                    ([emotion, b64]) => (
                      <div key={emotion}>
                        <p className="mb-1 text-center text-xs font-medium capitalize">
                          {emotion}
                        </p>
                        <SpectrogramViewer base64Image={b64} alt={`${emotion} spectrogram`} />
                      </div>
                    )
                  )}
                </div>
              ) : null}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
