"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type SpeedPoint = { time_s: number; speed_kmh: number };

type Props = {
  data: SpeedPoint[];
  sprintThreshold?: number;
};

export function SpeedChart({ data, sprintThreshold = 25 }: Props) {
  if (!data.length) {
    return (
      <div className="grid h-56 place-items-center rounded-xl border border-white/[0.06] bg-[#09090f] text-sm text-[#6b7a99]">
        Speed timeline will appear when track data is available.
      </div>
    );
  }

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="rgba(6,182,212,0.07)" strokeDasharray="4 4" />
          <XAxis
            dataKey="time_s"
            tick={{ fill: "#6b7a99", fontSize: 11 }}
            tickFormatter={(value: number) => `${value.toFixed(0)}s`}
            stroke="rgba(255,255,255,0.06)"
          />
          <YAxis
            tick={{ fill: "#6b7a99", fontSize: 11 }}
            stroke="rgba(255,255,255,0.06)"
            unit=" km/h"
          />
          <Tooltip
            contentStyle={{
              background: "#0d0d17",
              border: "1px solid rgba(6,182,212,0.25)",
              borderRadius: 10,
              color: "#eef2ff",
              boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
            }}
            labelFormatter={(value) => `${Number(value).toFixed(1)}s`}
            formatter={(value) => [`${Number(value).toFixed(1)} km/h`, "Speed"]}
          />
          <ReferenceLine
            y={sprintThreshold}
            stroke="#f59e0b"
            strokeDasharray="6 4"
            label={{ value: "Sprint threshold", fill: "#f59e0b", fontSize: 10 }}
          />
          <Line
            type="monotone"
            dataKey="speed_kmh"
            stroke="#06b6d4"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#22d3ee", stroke: "rgba(6,182,212,0.3)", strokeWidth: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ConfidenceDot({ score }: { score: number }) {
  const tone =
    score >= 0.7
      ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]"
      : score >= 0.4
        ? "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.5)]"
        : "bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.5)]";
  const label = score >= 0.7 ? "High" : score >= 0.4 ? "Medium" : "Low";
  return (
    <span className="inline-flex items-center gap-2 text-xs font-medium text-[#6b7a99]">
      <span className={`h-2 w-2 rounded-full ${tone}`} />
      {label} confidence
    </span>
  );
}
