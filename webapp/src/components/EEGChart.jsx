/**
 * EEGChart.jsx
 * ------------
 * Real-time scrolling EEG waveform chart (8 channels).
 *
 * Uses recharts LineChart to display the last 256 data points for each of the
 * 8 EEG channels simultaneously.
 *
 * Props:
 *  buffer {Array<number[]>} - Array of EEG samples; each sample is an array of
 *                             8 channel values.
 */

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

/** Distinct colours for the 8 EEG channels */
const CHANNEL_COLORS = [
  '#2196f3', // ch1 — blue
  '#4caf50', // ch2 — green
  '#f44336', // ch3 — red
  '#ff9800', // ch4 — orange
  '#9c27b0', // ch5 — purple
  '#00bcd4', // ch6 — cyan
  '#ffeb3b', // ch7 — yellow
  '#e91e63', // ch8 — pink
];

const N_CHANNELS = 8;

export default function EEGChart({ buffer }) {
  // Convert buffer array into recharts-compatible object array
  const chartData = useMemo(
    () =>
      buffer.map((sample, idx) => {
        const point = { t: idx };
        for (let ch = 0; ch < N_CHANNELS; ch++) {
          point[`ch${ch + 1}`] = sample[ch] ?? 0;
        }
        return point;
      }),
    [buffer]
  );

  return (
    <div style={styles.card}>
      <h3 style={styles.title}>📈 Live EEG Waveform</h3>
      <p style={styles.subtitle}>8 channels — last 256 samples</p>

      <ResponsiveContainer width="100%" height={320}>
        <LineChart
          data={chartData}
          margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
          <XAxis
            dataKey="t"
            tick={{ fill: '#666', fontSize: 11 }}
            label={{ value: 'sample', position: 'insideBottomRight', offset: 0, fill: '#666', fontSize: 11 }}
          />
          <YAxis tick={{ fill: '#666', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#13131f', border: '1px solid #2a2a3a', fontSize: 12 }}
            labelStyle={{ color: '#aaa' }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            formatter={(value) => (
              <span style={{ color: '#ccc' }}>{value}</span>
            )}
          />
          {Array.from({ length: N_CHANNELS }, (_, ch) => (
            <Line
              key={`ch${ch + 1}`}
              type="monotone"
              dataKey={`ch${ch + 1}`}
              stroke={CHANNEL_COLORS[ch]}
              dot={false}
              strokeWidth={1.2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const styles = {
  card: {
    background: '#13131f',
    borderRadius: '12px',
    padding: '20px',
    flex: 1,
  },
  title: {
    margin: '0 0 2px 0',
    fontSize: '1rem',
    fontWeight: 600,
  },
  subtitle: {
    margin: '0 0 12px 0',
    fontSize: '0.8rem',
    color: '#666',
  },
};
