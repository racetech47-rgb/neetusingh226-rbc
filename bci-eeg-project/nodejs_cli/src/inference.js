/**
 * src/inference.js
 * -----------------
 * ONNX-based inference module for the BCI EEG Brain State Classifier.
 *
 * Loads the exported ONNX model and runs inference on a feature vector.
 * Simulates a random feature vector when no real EEG data is provided.
 *
 * Exports:
 *   predictBrainState(features?) → Promise<{ state: string, confidence: number }>
 */

import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Path to the ONNX model (relative to this file)
const MODEL_PATH = path.resolve(__dirname, "../../export/bci_model.onnx");

// Human-readable class names matching Python label order (0–4)
const CLASS_NAMES = ["focus", "relax", "stress", "sleep", "meditation"];

// Expected feature vector length: 8 channels × 3 bands = 24 features
const FEATURE_DIM = 24;

// Lazy-loaded ONNX session (created once, then reused)
let _session = null;

/**
 * Load (or reuse) the ONNX inference session.
 *
 * @returns {Promise<import("onnxruntime-node").InferenceSession>}
 */
async function _getSession() {
  if (_session !== null) return _session;

  const ort = await import("onnxruntime-node");
  _session = await ort.InferenceSession.create(MODEL_PATH);
  return _session;
}

/**
 * Run inference on a feature vector and return the predicted brain state.
 *
 * @param {Float32Array|number[]|null} features
 *   Flat feature vector of length FEATURE_DIM (8 channels × 3 frequency bands).
 *   If null/undefined a random vector is generated for demonstration purposes.
 *
 * @returns {Promise<{ state: string, confidence: number }>}
 *   `state`      — predicted brain state name (lower-case).
 *   `confidence` — softmax probability of the winning class (0–1).
 */
export async function predictBrainState(features = null) {
  const ort = await import("onnxruntime-node");
  const session = await _getSession();

  // Build feature tensor
  let featureData;
  if (features === null || features === undefined) {
    // Simulate a random normalised feature vector
    featureData = Float32Array.from(
      { length: FEATURE_DIM },
      () => (Math.random() * 4 - 2)   // values in [-2, 2) like StandardScaler output
    );
  } else {
    featureData = features instanceof Float32Array
      ? features
      : new Float32Array(features);
  }

  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor("float32", featureData, [1, FEATURE_DIM]);
  const feeds = { [inputName]: tensor };

  const results = await session.run(feeds);

  // The output is a softmax probability vector
  const outputName = session.outputNames[0];
  const probs = Array.from(results[outputName].data);

  const bestIdx = probs.indexOf(Math.max(...probs));
  const state = CLASS_NAMES[bestIdx] ?? `class_${bestIdx}`;
  const confidence = probs[bestIdx];

  return { state, confidence };
}
