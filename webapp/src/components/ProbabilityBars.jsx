/**
 * ProbabilityBars.jsx
 * -------------------
 * Horizontal bar chart showing the probability for each brain state.
 *
 * Props:
 *  probs {Object} - Map of state name (lowercase) to probability 0-1
 *                   e.g. { focus: 0.94, relax: 0.03, ... }
 */

import React from 'react';

const STATES = [
  { key: 'focus',      label: 'FOCUS',      emoji: '🎯', color: '#2196f3' },
  { key: 'relax',      label: 'RELAX',      emoji: '😌', color: '#4caf50' },
  { key: 'stress',     label: 'STRESS',     emoji: '😰', color: '#f44336' },
  { key: 'sleep',      label: 'SLEEP',      emoji: '😴', color: '#9c27b0' },
  { key: 'meditation', label: 'MEDITATION', emoji: '🧘', color: '#ffeb3b' },
];

export default function ProbabilityBars({ probs }) {
  return (
    <div style={styles.card}>
      <h3 style={styles.title}>📊 State Probabilities</h3>
      {STATES.map(({ key, label, emoji, color }) => {
        const pct = ((probs[key] ?? 0) * 100).toFixed(1);
        return (
          <div key={key} style={styles.row}>
            <div style={styles.labelRow}>
              <span style={styles.emoji}>{emoji}</span>
              <span style={{ ...styles.label, color }}>{label}</span>
              <span style={styles.pct}>{pct}%</span>
            </div>
            <div style={styles.barBg}>
              <div
                style={{
                  ...styles.barFill,
                  width: `${Math.min(pct, 100)}%`,
                  background: color,
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

const styles = {
  card: {
    background: '#13131f',
    borderRadius: '12px',
    padding: '20px',
  },
  title: {
    margin: '0 0 14px 0',
    fontSize: '1rem',
    fontWeight: 600,
  },
  row: {
    marginBottom: '12px',
  },
  labelRow: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '4px',
    gap: '6px',
  },
  emoji: {
    fontSize: '1rem',
    lineHeight: 1,
  },
  label: {
    fontSize: '0.8rem',
    fontWeight: 600,
    letterSpacing: '0.06em',
    flex: 1,
  },
  pct: {
    fontSize: '0.8rem',
    color: '#aaa',
    minWidth: '42px',
    textAlign: 'right',
  },
  barBg: {
    background: '#2a2a3a',
    borderRadius: '4px',
    height: '6px',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: '4px',
  },
};
