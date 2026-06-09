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
      <div className="grid h-56 place-items-center rounded-lg border border-[#ffffff14] bg-[#0a0a0f] text-sm text-[#64748b]">
        Speed timeline will appear when track data is available.
      </div>
    );
  }

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#ffffff10" strokeDasharray="3 3" />
          <XAxis
            dataKey="time_s"
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickFormatter={(value: number) => `${value.toFixed(0)}s`}
            stroke="#ffffff14"
          />
          <YAxis
            tick={{ fill: "#64748b", fontSize: 11 }}
            stroke="#ffffff14"
            unit=" km/h"
          />
          <Tooltip
            contentStyle={{
              background: "#111118",
              border: "1px solid #ffffff14",
              borderRadius: 8,
              color: "#f1f5f9",
            }}
            labelFormatter={(value) => `${Number(value).toFixed(1)}s`}
            formatter={(value) => [`${Number(value).toFixed(1)} km/h`, "Speed"]}
          />
          <ReferenceLine
            y={sprintThreshold}
            stroke="#f59e0b"
            strokeDasharray="6 4"
            label={{ value: "Sprint threshold", fill: "#f59e0b", fontSize: 11 }}
          />
          <Line
            type="monotone"
            dataKey="speed_kmh"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#3b82f6" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ConfidenceDot({ score }: { score: number }) {
  const tone =
    score >= 0.7 ? "bg-[#10b981]" : score >= 0.4 ? "bg-[#f59e0b]" : "bg-red-500";
  const label = score >= 0.7 ? "High" : score >= 0.4 ? "Medium" : "Low";
  return (
    <span className="inline-flex items-center gap-2 text-sm text-[#64748b]">
      <span className={`h-2.5 w-2.5 rounded-full ${tone}`} />
      {label} confidence
    </span>
  );
}
