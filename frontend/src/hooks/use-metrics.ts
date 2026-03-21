"use client";

import { useQuery } from "@tanstack/react-query";
import { getMetrics, getHealth } from "@/lib/api";

export function useMetrics() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: getMetrics,
    retry: false,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30000,
  });
}
