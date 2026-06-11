# Benchmark Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-backed benchmark calibration, richer runtime/hardware coverage, prefill/decode estimator outputs, and actionable fit recommendations.

**Architecture:** Preserve the existing React/Vite app shape. Keep estimates in `src/lib/calculator.js`, static data in `src/data/*.json`, and UI wiring in `src/main.jsx`. Add no new runtime dependencies.

**Tech Stack:** React 19, Vite, Mantine, lucide-react, JSON data files.

---

### Task 1: Runtime And Estimator Fields

**Files:**
- Modify: `src/lib/calculator.js`

- [ ] Add runtime profiles for `tensorrt`, `mlc`, `tgi`, `lmdeploy`, and `exllamav3`.
- [ ] Split performance into `decodeTokSec`, `prefillTokSec`, `ttftMs`, `interTokenLatencyMs`, `maxResidentSequences`, and `queuePenaltyMs`.
- [ ] Keep existing aliases: `tokensPerSecond`, `msPerToken`, `timeToFirstTokenMs`, and `totalThroughput`.
- [ ] Verify by importing `calculateInference` in a Node script and checking all new keys are numeric for a known config.

Expected command:

```bash
node --input-type=module - <<'NODE'
import models from './src/data/models.json' with { type: 'json' };
import hardware from './src/data/hardware.json' with { type: 'json' };
import { calculateInference } from './src/lib/calculator.js';
const result = calculateInference({
  mode: 'inference',
  model: models.find((m) => m.slug === 'llama-3-1-8b'),
  hardware: hardware.find((h) => h.id === 'h100-sxm-80'),
  weightQuant: 'q4_k_m',
  kvQuant: 'fp16',
  trainingMethod: 'lora',
  numGpus: 1,
  batchSize: 1,
  sequenceLength: 4096,
  concurrentUsers: 1,
  offload: false,
  runtimeProfile: 'vllm'
});
console.log(['decodeTokSec','prefillTokSec','ttftMs','interTokenLatencyMs','maxResidentSequences'].map((key) => [key, Number.isFinite(result[key])]));
NODE
```

### Task 2: Data Additions

**Files:**
- Modify: `src/data/hardware.json`
- Modify: `src/data/benchmarks.json`

- [ ] Add high-confidence hardware records from research: H200 NVL, DGX B200, B300, DGX B300, GB200/GB300 rack/module, AMD MI300X/MI325X/MI350X/MI355X, AWS Inferentia/Trainium, Google TPU v5e/v5p/v6e/Ironwood, Positron Atlas, Cerebras CS-3, SambaNova SN40L.
- [ ] Add benchmark rows from runtime research for TensorRT/NIM, Spheron H100 runtime comparison, MLC H100 comparison, llama.cpp CUDA/LocalScore, and Ollama+MLX Qwen3.5.
- [ ] Validate no duplicate hardware ids and no benchmark hardware references are missing.

Expected command:

```bash
node - <<'NODE'
const fs = require('fs');
const hw = JSON.parse(fs.readFileSync('src/data/hardware.json', 'utf8'));
const bench = JSON.parse(fs.readFileSync('src/data/benchmarks.json', 'utf8'));
const ids = new Set();
for (const item of hw) {
  if (ids.has(item.id)) throw new Error(`duplicate hardware id ${item.id}`);
  ids.add(item.id);
}
const missing = bench.filter((row) => row.hardwareId && !ids.has(row.hardwareId)).map((row) => row.hardwareId);
if (missing.length) throw new Error(`missing hardware refs ${[...new Set(missing)].join(', ')}`);
console.log({ hardware: hw.length, benchmarks: bench.length });
NODE
```

### Task 3: Observed Benchmark Comparison UI

**Files:**
- Modify: `src/main.jsx`
- Modify: `src/styles.css`

- [ ] Rename displayed “Generation Speed” to “Decode Speed”.
- [ ] Add “Prefill Speed” to the metric list.
- [ ] Extend `ObservedBenchmarks` to show estimated decode, observed decode, delta, prefill/TTFT, and peak memory.
- [ ] Keep rows linked to source URLs and label confidence.
- [ ] Verify MiniMax M2.7 + M5 Max 128GB shows observed oMLX rows and deltas.

### Task 4: Fit Recommendations

**Files:**
- Modify: `src/main.jsx`
- Modify: `src/styles.css`

- [ ] Add `getFitRecommendations` helper using current model, hardware, quantization rows, context, GPU count, offload state, and hardware profiles.
- [ ] Add `RecommendationPanel` under the hero result when the run is over budget.
- [ ] Provide action buttons for quant switch, context reduction, GPU increase, hardware switch, and enabling offload where applicable.
- [ ] Verify over-budget MiniMax M2.7 FP16 on M5 Max 128GB shows useful actions.

### Task 5: Verification

**Files:**
- No code changes unless verification finds issues.

- [ ] Run JSON validation.
- [ ] Run `npm run build`.
- [ ] Start/confirm Vite server.
- [ ] Use in-app browser to verify app loads without console errors.
- [ ] Exercise one benchmark panel flow and one recommendation flow.

---

## Plan Self-Review

- Every task has exact files and expected outcomes.
- Implementation is bounded and does not require new dependencies.
- Low-confidence benchmark claims are accepted only as source-labeled rows, not estimator defaults.
- Existing public API of `calculateInference` remains compatible.
