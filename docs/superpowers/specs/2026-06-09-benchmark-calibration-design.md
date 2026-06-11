# Benchmark Calibration Design

**Goal:** Upgrade the calculator from a single heuristic tok/sec estimate into a source-backed inference planning tool that distinguishes memory fit, decode speed, prefill speed, TTFT, observed benchmark evidence, and actionable fit recommendations.

**Research Inputs:** Four independent research tracks covered runtime benchmarks, hardware specs, estimator design, and current-app integration. The implementation uses high-confidence and medium-confidence rows only where the source is explicit enough to label provenance.

## Scope

This increment implements:

- Runtime profiles for TensorRT-LLM/NIM, MLC-LLM, TGI, LMDeploy, and ExLlamaV3 in addition to the existing MLX, llama.cpp, Ollama, vLLM, and SGLang profiles.
- Explicit estimator outputs for decode tok/sec, prefill tok/sec, TTFT, ITL, throughput, and resident sequence capacity. Existing `tokensPerSecond` remains a decode-speed alias.
- Observed benchmark comparison against current estimates, including nearest-run context labels and estimate-vs-observed deltas.
- Actionable over-budget recommendations: lower quantization, reduce context, add GPUs, choose larger fitting hardware, or enable offload.
- High-confidence hardware additions for current NVIDIA, AMD, AWS, Google, and accelerator systems where memory/bandwidth are known.
- Benchmark additions for TensorRT/NIM, vLLM, SGLang, MLC, llama.cpp, Ollama+MLX, and relevant MiniMax rows.

This increment does not implement:

- Full benchmark import/export UI.
- CSV parsing.
- Complete migration of quant metadata into a standalone schema.
- Vendor-only unverified performance claims as estimator defaults.

## Architecture

The existing boundaries stay intact:

- `src/lib/calculator.js` remains the estimate engine.
- `src/data/*.json` remains the source of static model, hardware, and benchmark data.
- `src/main.jsx` remains the UI shell, with added helper functions for recommendations and benchmark deltas.

The estimator returns richer fields but preserves current callers:

- `tokensPerSecond` aliases `decodeTokSec`.
- `timeToFirstTokenMs` aliases `ttftMs`.
- `totalThroughput` stays derived from decode speed and concurrency.

Observed benchmarks remain separate from estimates. The UI labels them as observed/reported/vendor so the user can evaluate source quality.

## Data Rules

Hardware records require:

- `id`, `vendor`, `name`, `memoryGb`, `bandwidthGbps`, `powerW`, `priceHour`, `type`.

Benchmark records require:

- `id`, `modelSlug`, `hardwareId`, `engine`, `engineLabel`, `generationTokSec`, `sourceName`, `sourceUrl`, `confidence`.

Optional benchmark fields:

- `prefillTokSec`, `ttftMs`, `interTokenLatencyMs`, `peakMemoryGb`, `contextTokens`, `batchSize`, `concurrency`, `inputTokens`, `outputTokens`, `maxOutputTokens`, `powerW`, `notes`.

Low-confidence claims may be stored only if marked `confidence: "reported"` or weaker and are never used silently as calibration constants.

## UX

Primary result card changes:

- Rename “Generation Speed” to “Decode Speed”.
- Add “Prefill Speed”.
- Keep TTFT and throughput, but report “Unavailable” when over budget.

Observed benchmark card changes:

- Show engine, run shape, observed decode, estimated decode, delta, prefill/TTFT, and memory.
- Highlight the active runtime when it matches.
- Keep source links clickable.

Recommendation card:

- Appears only when `result.isRunnable === false`.
- Prioritizes recommendations by preserving quality first, then reducing memory pressure.
- Actions that can be applied locally use buttons.

## Testing

Verification must include:

- JSON parse and missing-reference checks.
- `npm run build`.
- Browser smoke test on the running Vite app.
- One benchmark-matching flow, preferably MiniMax M2.7 + M5 Max 128GB.
- One over-budget recommendation flow, preferably MiniMax M2.7 FP16 on M5 Max 128GB or RTX PRO 6000.

## Self-Review

- No placeholders remain.
- Scope is bounded to data, estimator fields, benchmark comparison, and fit recommendations.
- Hardware/vendor claims remain explicitly source-labeled.
- Backward compatibility is preserved for existing UI fields.
