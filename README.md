# LLM VRAM & Performance Calculator

Source-backed calculator for local and server-side LLM inference planning.

It estimates:

- VRAM allocation by weights, activations, KV cache, and runtime overhead
- Decode speed, prefill speed, TTFT, throughput, cost, and power
- Quantization fit across common weight formats
- Hardware/model comparison profiles
- Evidence and benchmark provenance for model specs and performance assumptions

## Development

```bash
npm install
npm run dev
```

## Verification

```bash
npm test
npm run build
```

## Deployment

The app is deployed to GitHub Pages by `.github/workflows/pages.yml` on pushes to `main`.

