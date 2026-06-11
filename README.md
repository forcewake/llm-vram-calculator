# LLM Inference VRAM & Performance Calculator

Source-backed web calculator for planning local and server-side LLM inference runs.

- Live app: https://forcewake.github.io/llm-vram-calculator/
- Repository: https://github.com/forcewake/llm-vram-calculator
- Stack: React 19, Vite 7, Mantine 9, lucide-react
- Runtime: static client app, no backend required

## What It Does

The calculator estimates whether a selected model and runtime profile can fit on a selected GPU, Apple unified-memory device, or multi-GPU setup. It is designed for practical inference planning, not just raw parameter counting.

It currently covers:

- VRAM allocation across model weights, KV cache, activations, and runtime overhead
- Decode speed, prefill speed, time to first token, total throughput, and latency
- Quantization fit tables across FP32, FP16/BF16, FP8, NVFP, Q8, Q7, Q6, Q5, Q4, Q3, and Q2-style formats
- Runtime profiles for generic inference, MLX/Metal, llama.cpp, Ollama, vLLM, SGLang, TensorRT-LLM/NIM, MLC-LLM, Hugging Face TGI, LMDeploy, and ExLlamaV3
- Compare Profiles view with cloneable profiles and table/card layouts
- Evidence-backed model, hardware, benchmark, and methodology data

## Data Coverage

The checked-in data set currently includes:

- 185 model profiles in `src/data/models.json`
- 96 hardware profiles in `src/data/hardware.json`
- 38 observed benchmark records in `src/data/benchmarks.json`
- 13 methodology/performance sources in `src/data/performance-sources.json`

Every model and hardware record is expected to include source URLs and evidence entries. Benchmark records link a model, hardware profile, engine, quantization, context length, speed, and source.

## Data Files

`src/data/models.json`

Model catalog with provider, family, architecture, parameter count, context length, attention/KV metadata, estimated VRAM baselines, API-cost placeholders where known, and evidence. Each model also has a `performance` block used by the estimator.

`src/data/hardware.json`

Hardware catalog with memory, bandwidth, power, hourly cost estimate, device class, and evidence.

`src/data/benchmarks.json`

Observed benchmark samples. These records are used to ground speed expectations for specific model, hardware, engine, quantization, and context combinations.

`src/data/performance-sources.json`

Methodology references for metrics such as TTFT, decode throughput, prefill throughput, serving runtime behavior, and inference benchmarking assumptions.

## Evidence Rules

The data provenance test enforces evidence quality. Each evidence item must include:

- `kind`: one of `official`, `docs`, `datasheet`, `paper`, `weights`, `benchmark`, `github`, `reddit`, `community`, or `pricing`
- `url`: a valid HTTP(S) URL
- `label`: readable source label
- `claim`: specific statement supported by the source
- `quote`: short exact quote, capped at 25 words
- `quoteUrl`: source URL or direct quote URL
- `accessedAt`: `YYYY-MM-DD`

Field-level evidence is supported on model records through `fieldEvidence`, and the test requires broad field-level evidence coverage across the catalog.

## Estimation Model

Core calculations live in `src/lib/calculator.js`.

The inference calculator estimates:

- Available memory: hardware memory multiplied by GPU count and runtime usable-memory ratio
- Model weights: parameter count multiplied by the selected quantization byte width
- KV cache: model KV dimensions, context length, batch size, concurrent users, and KV quantization
- Activations: model size, sequence length, and batch size
- Framework overhead: runtime-dependent overhead based on model and hardware scale
- Fit status: total required memory versus runtime-adjusted usable memory
- Decode speed: bandwidth, active parameter count, quantization penalty, runtime multiplier, pressure penalty, and offload penalty
- Prefill speed: bandwidth, model scale, runtime prefill multiplier, context penalty, and pressure/offload penalties
- TTFT: prefill time, runtime multiplier, memory pressure, offload penalty, and queueing

Mixture-of-experts models can use active-parameter overrides where the full parameter count is not the right proxy for token-by-token compute.

## Runtime Profiles

Runtime profiles calibrate usable memory, speed, prefill behavior, queueing, and overhead. The app can auto-select a likely runtime profile based on hardware, or the user can select a profile manually.

Examples:

- Apple unified memory defaults toward MLX/Metal behavior
- Local GGUF-style flows can use llama.cpp or Ollama
- Server deployments can use vLLM, SGLang, TensorRT-LLM/NIM, TGI, LMDeploy, or MLC-LLM

These are estimates. Real results still depend on model file format, kernels, batch scheduler, prompt shape, driver/runtime version, thermal limits, and exact quantization implementation.

## Local Development

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

The dev server binds to `127.0.0.1` by default. Vite will print the exact port.

Build the production bundle:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

## Verification

Run all project tests:

```bash
npm test
```

Run the data provenance test directly:

```bash
npm run test:data
```

Run the production build:

```bash
npm run build
```

The current verification gate is intentionally data-heavy: it validates source URLs, evidence shape, quote limits, field-level evidence, benchmark links, performance method references, and catalog consistency.

## GitHub Pages Deployment

Deployment is handled by `.github/workflows/pages.yml`.

On every push to `main`, GitHub Actions runs:

1. `npm ci`
2. `npm run test:data`
3. `npm run build`
4. Uploads `dist`
5. Deploys to GitHub Pages

The Vite base path is configured in `vite.config.js`:

```js
base: process.env.GITHUB_ACTIONS ? '/llm-vram-calculator/' : '/'
```

This keeps local development at `/` while publishing correctly under the GitHub Pages project path.

## Repository Layout

```text
.
├── .github/workflows/pages.yml
├── docs/superpowers/
├── index.html
├── package.json
├── src/
│   ├── data/
│   │   ├── benchmarks.json
│   │   ├── hardware.json
│   │   ├── models.json
│   │   └── performance-sources.json
│   ├── lib/calculator.js
│   ├── main.jsx
│   └── styles.css
├── tests/data-provenance.test.mjs
└── vite.config.js
```

## Maintenance Workflow

When adding or changing model data:

1. Add or update the model record in `src/data/models.json`.
2. Add direct source URLs and evidence entries.
3. Add field-level evidence when changing numeric model metadata.
4. Add observed benchmark records in `src/data/benchmarks.json` when available.
5. Link methodology assumptions through `performance.methodEvidenceIds`.
6. Run `npm run test:data`.
7. Run `npm run build`.

When adding hardware:

1. Add the device to `src/data/hardware.json`.
2. Include memory, bandwidth, power, cost estimate where available, and device type.
3. Prefer official vendor sources or datasheets.
4. Run `npm run test:data`.

When changing formulas:

1. Update `src/lib/calculator.js`.
2. Re-check representative dense, MoE, small, large, Apple unified-memory, and datacenter GPU scenarios.
3. Add or update methodology sources when the assumption changes.
4. Run `npm test` and `npm run build`.

## Limitations

The calculator is a planning tool, not a hardware guarantee. Estimates can differ from local measurements because runtimes vary in kernels, memory allocators, prompt batching, cache layout, speculative decoding, quantization packing, GPU clocks, thermal behavior, and OS memory pressure.

Observed benchmark data is preferred where available. Otherwise the app falls back to transparent, source-linked estimation heuristics.

## License

No license file has been added yet. Until a license is selected, normal copyright restrictions apply.
