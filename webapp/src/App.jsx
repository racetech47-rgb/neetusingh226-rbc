/**
 * App.jsx
 * -------
 * Root application component.
 *
 * Connects to the BCI API WebSocket endpoint (ws://localhost:8000/ws) and
 * streams real-time brain-state predictions.  Auto-reconnects on disconnect.
 *
 * Features:
 *  - Animated brain-state badge (BrainStateDisplay)
 *  - Live 8-channel EEG waveform (EEGChart)
 *  - Per-state probability bars (ProbabilityBars)
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import BrainStateDisplay from './components/BrainStateDisplay';
import EEGChart from './components/EEGChart';
import ProbabilityBars from './components/ProbabilityBars';

const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_DELAY_MS = 2000;

/** Colour map per brain state */
export const STATE_COLORS = {
  FOCUS:      '#2196f3', // 🔵 blue
  RELAX:      '#4caf50', // 🟢 green
  STRESS:     '#f44336', // 🔴 red
  SLEEP:      '#9c27b0', // 🟣 purple
  MEDITATION: '#ffeb3b', // 🟡 yellow
};

const INITIAL_PROBS = {
  focus: 0,
  relax: 0,
  stress: 0,
  sleep: 0,
  meditation: 0,
};

export default function App() {
  const [connected, setConnected]       = useState(false);
  const [brainState, setBrainState]     = useState('--');
  const [confidence, setConfidence]     = useState(0);
  const [allProbs, setAllProbs]         = useState(INITIAL_PROBS);
  const [eegBuffer, setEegBuffer]       = useState([]);

  const wsRef              = useRef(null);
  const reconnectTimer     = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setBrainState(data.state ?? '--');
        setConfidence((data.confidence ?? 0) * 100);
        setAllProbs(data.all_probs ?? INITIAL_PROBS);

        if (Array.isArray(data.eeg_sample)) {
          setEegBuffer((prev) => {
            const next = [...prev, data.eeg_sample];
            // Keep the last 256 data points
            return next.length > 256 ? next.slice(next.length - 256) : next;
          });
        }
      } catch (err) {
        // Malformed message — log for debugging and ignore
        console.warn('Failed to parse WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const stateColor = STATE_COLORS[brainState] ?? '#888';

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>🧠 BCI Live Dashboard</h1>
        <span style={{ ...styles.badge, background: connected ? '#4caf50' : '#f44336' }}>
          {connected ? '● Connected' : '○ Disconnected'}
        </span>
      </header>

      {/* Main grid */}
      <div style={styles.grid}>
        {/* Left column: brain state + probabilities */}
        <div style={styles.leftCol}>
          <BrainStateDisplay
            state={brainState}
            confidence={confidence}
            color={stateColor}
          />
          <ProbabilityBars probs={allProbs} />
        </div>

        {/* Right column: EEG chart */}
        <div style={styles.rightCol}>
          <EEGChart buffer={eegBuffer} />
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    background: '#0a0a0f',
    color: '#e0e0e0',
    padding: '16px',
    boxSizing: 'border-box',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '24px',
    borderBottom: '1px solid #2a2a3a',
    paddingBottom: '12px',
  },
  title: {
    margin: 0,
    fontSize: '1.6rem',
    letterSpacing: '0.03em',
  },
  badge: {
    padding: '4px 12px',
    borderRadius: '12px',
    fontSize: '0.85rem',
    fontWeight: 600,
    color: '#fff',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '320px 1fr',
    gap: '24px',
  },
  leftCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  rightCol: {
    display: 'flex',
    flexDirection: 'column',
  },
};
