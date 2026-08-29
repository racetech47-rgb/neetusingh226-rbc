/**
 * BrainStateDisplay.jsx
 * ---------------------
 * Large animated brain-state badge.
 *
 * Props:
 *  state      {string}  - Current brain state (e.g. "FOCUS")
 *  confidence {number}  - Confidence percentage 0-100
 *  color      {string}  - Accent colour for the state
 */

import React, { useEffect, useRef } from 'react';

const STATE_EMOJI = {
  FOCUS:      '🎯',
  RELAX:      '😌',
  STRESS:     '😰',
  SLEEP:      '😴',
  MEDITATION: '🧘',
};

export default function BrainStateDisplay({ state, confidence, color }) {
  const prevStateRef = useRef(state);
  const badgeRef     = useRef(null);

  // Trigger pulse animation when state changes
  useEffect(() => {
    if (state !== prevStateRef.current && badgeRef.current) {
      prevStateRef.current = state;
      const el = badgeRef.current;
      el.classList.remove('pulse');
      // Force reflow so re-adding the class restarts the animation
      void el.offsetWidth;
      el.classList.add('pulse');
    }
  }, [state]);

  const emoji = STATE_EMOJI[state] ?? '🧠';

  return (
    <div style={styles.card}>
      <style>{`
        @keyframes pulseAnim {
          0%   { box-shadow: 0 0 0 0 ${color}88; }
          50%  { box-shadow: 0 0 24px 12px ${color}44; }
          100% { box-shadow: 0 0 0 0 ${color}00; }
        }
        .pulse { animation: pulseAnim 0.6s ease-out; }
      `}</style>

      <div ref={badgeRef} style={{ ...styles.badge, borderColor: color }}>
        <span style={styles.emoji}>{emoji}</span>
        <span style={{ ...styles.stateLabel, color }}>{state}</span>
      </div>

      <div style={styles.confidenceRow}>
        <span style={styles.confidenceLabel}>Confidence</span>
        <span style={{ ...styles.confidenceValue, color }}>
          {confidence.toFixed(1)}%
        </span>
      </div>

      {/* Confidence bar */}
      <div style={styles.barBg}>
        <div
          style={{
            ...styles.barFill,
            width: `${Math.min(confidence, 100)}%`,
            background: color,
            transition: 'width 0.3s ease',
          }}
        />
      </div>
    </div>
  );
}

const styles = {
  card: {
    background: '#13131f',
    borderRadius: '12px',
    padding: '20px',
    textAlign: 'center',
  },
  badge: {
    display: 'inline-flex',
    flexDirection: 'column',
    alignItems: 'center',
    border: '2px solid',
    borderRadius: '16px',
    padding: '16px 28px',
    marginBottom: '16px',
    transition: 'border-color 0.3s ease',
  },
  emoji: {
    fontSize: '3rem',
    lineHeight: 1.2,
  },
  stateLabel: {
    fontSize: '1.4rem',
    fontWeight: 700,
    letterSpacing: '0.1em',
    marginTop: '6px',
    transition: 'color 0.3s ease',
  },
  confidenceRow: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '6px',
    fontSize: '0.9rem',
  },
  confidenceLabel: {
    color: '#888',
  },
  confidenceValue: {
    fontWeight: 700,
    fontSize: '1rem',
  },
  barBg: {
    background: '#2a2a3a',
    borderRadius: '4px',
    height: '8px',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: '4px',
  },
};
