#!/usr/bin/env node
/**
 * bin/bci.js
 * ----------
 * ESM Node.js CLI entry point for the BCI EEG Brain State Classifier.
 *
 * Commands:
 *   bci predict   — load ONNX model, run inference on simulated EEG, print result
 *   bci info      — print model input/output shape and available brain states
 *
 * Usage:
 *   node bin/bci.js predict
 *   node bin/bci.js info
 */

import { program } from "commander";
import chalk from "chalk";
import { predictBrainState } from "../src/inference.js";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ------------------------------------------------------------------ //
// CLI definition                                                       //
// ------------------------------------------------------------------ //

program
  .name("bci")
  .description(chalk.cyan("🧠 BCI EEG Brain State Classifier CLI"))
  .version("1.0.0");

// ---- predict ----
program
  .command("predict")
  .description("Run inference on simulated EEG data and print the brain state")
  .option("-n, --samples <number>", "Number of simulated feature vectors to classify", "1")
  .action(async (options) => {
    const nSamples = parseInt(options.samples, 10);
    console.log(chalk.cyan(`\n🧠 BCI Brain State Classifier — predicting ${nSamples} sample(s)…\n`));

    for (let i = 0; i < nSamples; i++) {
      try {
        const { state, confidence } = await predictBrainState();
        const confPct = (confidence * 100).toFixed(1);
        const stateColored = chalk.bold.green(state.toUpperCase());
        const confColored = chalk.yellow(`${confPct}%`);
        console.log(`  [${i + 1}] 🧠 Brain State: ${stateColored}  (confidence: ${confColored})`);
      } catch (err) {
        console.error(chalk.red(`  [${i + 1}] Error: ${err.message}`));
      }
    }
    console.log();
  });

// ---- info ----
program
  .command("info")
  .description("Print model info: input shape, output classes, and brain state descriptions")
  .action(async () => {
    console.log(chalk.cyan("\n🧠 BCI Model Info\n"));

    const modelPath = path.resolve(__dirname, "../../export/bci_model.onnx");

    try {
      const ort = await import("onnxruntime-node");
      const session = await ort.InferenceSession.create(modelPath);

      console.log(chalk.bold("  Input tensors:"));
      for (const name of session.inputNames) {
        const meta = session.inputMetadata?.[name];
        console.log(`    • ${chalk.green(name)}${meta ? `  dims=${JSON.stringify(meta.dims)}` : ""}`);
      }

      console.log(chalk.bold("\n  Output tensors:"));
      for (const name of session.outputNames) {
        console.log(`    • ${chalk.green(name)}`);
      }
    } catch (err) {
      console.log(
        chalk.yellow(`  (Could not load ONNX model at ${modelPath}: ${err.message})\n`) +
        chalk.dim("  Run `python main.py --mode export-onnx` to generate the model first.")
      );
    }

    console.log(chalk.bold("\n  Brain states (output classes):"));
    const states = [
      { id: 0, name: "FOCUS",      desc: "Active thinking, alertness — dominant beta (13–30 Hz)" },
      { id: 1, name: "RELAX",      desc: "Calm, idle state — dominant alpha (8–13 Hz)" },
      { id: 2, name: "STRESS",     desc: "Heightened arousal — high beta + gamma (25–100 Hz)" },
      { id: 3, name: "SLEEP",      desc: "Drowsiness / deep sleep — delta + theta (0.5–8 Hz)" },
      { id: 4, name: "MEDITATION", desc: "Focused calm — theta + alpha (4–13 Hz)" },
    ];
    for (const s of states) {
      console.log(`    ${chalk.bold.cyan(s.id)}: ${chalk.green(s.name.padEnd(12))} — ${chalk.dim(s.desc)}`);
    }
    console.log();
  });

program.parse();
