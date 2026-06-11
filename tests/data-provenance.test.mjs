import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const urlPattern = /^https?:\/\/[^\s]+$/;
const allowedEvidenceKinds = new Set(['official', 'docs', 'datasheet', 'paper', 'weights', 'benchmark', 'github', 'reddit', 'community', 'pricing']);

async function readJson(path) {
  return JSON.parse(await readFile(new URL(`../${path}`, import.meta.url), 'utf8'));
}

function flattenUrls(value) {
  if (!value) return [];
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap(flattenUrls);
  if (typeof value === 'object') {
    if (typeof value.url === 'string') return [value.url];
    return Object.values(value).flatMap(flattenUrls);
  }
  return [];
}

function assertEvidenceArray(recordType, record) {
  assert.ok(Array.isArray(record.evidence), `${recordType} ${record.id || record.slug} must have evidence[]`);
  assert.ok(record.evidence.length > 0, `${recordType} ${record.id || record.slug} must include at least one evidence item`);

  for (const item of record.evidence) {
    assert.ok(item && typeof item === 'object', `${recordType} ${record.id || record.slug} evidence item must be an object`);
    assert.ok(allowedEvidenceKinds.has(item.kind), `${recordType} ${record.id || record.slug} evidence kind ${item.kind} is not allowed`);
    assert.match(item.url, urlPattern, `${recordType} ${record.id || record.slug} evidence URL is invalid`);
    assert.ok(typeof item.label === 'string' && item.label.trim(), `${recordType} ${record.id || record.slug} evidence item needs a label`);
    assert.ok(typeof item.claim === 'string' && item.claim.trim().length >= 12, `${recordType} ${record.id || record.slug} evidence item needs a specific claim`);
    assert.ok(typeof item.quote === 'string' && item.quote.trim().length >= 4, `${recordType} ${record.id || record.slug} evidence item needs an exact quote`);
    assert.ok(!/Please wait for verification|Blocked/i.test(item.quote), `${recordType} ${record.id || record.slug} evidence quote is a placeholder, not a citation`);
    assert.ok(item.quote.split(/\s+/).length <= 25, `${recordType} ${record.id || record.slug} evidence quote must stay short`);
    assert.match(item.quoteUrl || item.url, urlPattern, `${recordType} ${record.id || record.slug} evidence quoteUrl is invalid`);
    assert.match(item.accessedAt, /^\d{4}-\d{2}-\d{2}$/, `${recordType} ${record.id || record.slug} evidence accessedAt must be YYYY-MM-DD`);
    assert.notEqual(item.quote.trim(), item.label.trim(), `${recordType} ${record.id || record.slug} evidence quote must not just repeat the label`);
    assert.notEqual(item.quote.trim(), item.claim.trim(), `${recordType} ${record.id || record.slug} evidence quote must not just repeat the claim`);
  }
}

const models = await readJson('src/data/models.json');
const hardware = await readJson('src/data/hardware.json');
const benchmarks = await readJson('src/data/benchmarks.json');
const performanceSources = await readJson('src/data/performance-sources.json');

const modelSlugs = new Set(models.map((model) => model.slug));
const hardwareIds = new Set(hardware.map((profile) => profile.id));

for (const model of models) {
  const urls = flattenUrls(model.source);
  assert.ok(urls.length > 0, `model ${model.slug} must have source URLs`);
  for (const url of urls.filter(Boolean)) assert.match(url, urlPattern, `model ${model.slug} has invalid source URL`);
  assertEvidenceArray('model', model);
  assert.ok(
    model.evidence.some((item) => item.claim.includes(model.name) || item.claim.includes(model.provider)),
    `model ${model.slug} evidence claims must identify the model`
  );
  if (model.fieldEvidence) {
    for (const [field, item] of Object.entries(model.fieldEvidence)) {
      assert.match(item.url, urlPattern, `model ${model.slug} fieldEvidence.${field} URL is invalid`);
      assert.ok(typeof item.claim === 'string' && item.claim.includes(model.name), `model ${model.slug} fieldEvidence.${field} needs a model-specific claim`);
      assert.ok(typeof item.quote === 'string' && item.quote.trim().length >= 4, `model ${model.slug} fieldEvidence.${field} needs an exact quote`);
      assert.ok(item.quote.split(/\s+/).length <= 25, `model ${model.slug} fieldEvidence.${field} quote must stay short`);
      assert.match(item.accessedAt, /^\d{4}-\d{2}-\d{2}$/, `model ${model.slug} fieldEvidence.${field} accessedAt must be YYYY-MM-DD`);
      if (typeof model[field] === 'number') {
        assert.ok(
          item.quote.includes(String(model[field])),
          `model ${model.slug} fieldEvidence.${field} quote must include the current value ${model[field]}`
        );
      }
    }
  }
  assert.ok(model.performance && typeof model.performance === 'object', `model ${model.slug} needs performance profile`);
  assert.ok(Number.isFinite(model.performance.computeParamsB) && model.performance.computeParamsB > 0, `model ${model.slug} needs computeParamsB`);
  assert.ok(Number.isFinite(model.performance.decodeWorkFactor) && model.performance.decodeWorkFactor > 0, `model ${model.slug} needs decodeWorkFactor`);
  assert.ok(Number.isFinite(model.performance.prefillWorkFactor) && model.performance.prefillWorkFactor > 0, `model ${model.slug} needs prefillWorkFactor`);
  assert.ok(Number.isFinite(model.performance.kvCacheGbPer1kFp16) && model.performance.kvCacheGbPer1kFp16 >= 0, `model ${model.slug} needs kvCacheGbPer1kFp16`);
  assert.ok(['very-fast', 'fast', 'balanced', 'slow', 'very-slow'].includes(model.performance.decodeSpeedClass), `model ${model.slug} has invalid decodeSpeedClass`);
  assert.ok(['low', 'moderate', 'high', 'extreme'].includes(model.performance.ttftRiskClass), `model ${model.slug} has invalid ttftRiskClass`);
  assert.ok(Array.isArray(model.performance.methodEvidenceIds) && model.performance.methodEvidenceIds.length > 0, `model ${model.slug} needs methodEvidenceIds`);
}

const modelsWithFieldEvidence = models.filter((model) => model.fieldEvidence && Object.keys(model.fieldEvidence).length > 0);
assert.ok(modelsWithFieldEvidence.length >= 100, `expected at least 100 models with field-level evidence, got ${modelsWithFieldEvidence.length}`);

for (const profile of hardware) {
  assertEvidenceArray('hardware', profile);
  assert.ok(
    profile.evidence.some((item) => item.claim.includes(profile.name) || item.claim.includes(profile.vendor)),
    `hardware ${profile.id} evidence claims must identify the hardware`
  );
  assert.ok(
    profile.evidence.every((item) => item.kind !== 'official' || item.sourceTier === 'primary'),
    `hardware ${profile.id} official evidence cannot be community tier`
  );
}

for (const benchmark of benchmarks) {
  assert.ok(modelSlugs.has(benchmark.modelSlug), `benchmark ${benchmark.id} references missing model ${benchmark.modelSlug}`);
  assert.ok(hardwareIds.has(benchmark.hardwareId), `benchmark ${benchmark.id} references missing hardware ${benchmark.hardwareId}`);
  assert.match(benchmark.sourceUrl, urlPattern, `benchmark ${benchmark.id} needs a valid sourceUrl`);
  assert.ok(Number.isFinite(benchmark.generationTokSec) && benchmark.generationTokSec > 0, `benchmark ${benchmark.id} needs decode/generation tok/sec`);
  assert.ok(benchmark.prefillTokSec === undefined || (Number.isFinite(benchmark.prefillTokSec) && benchmark.prefillTokSec > 0), `benchmark ${benchmark.id} has invalid prefillTokSec`);
  assert.ok(benchmark.ttftMs === undefined || (Number.isFinite(benchmark.ttftMs) && benchmark.ttftMs > 0), `benchmark ${benchmark.id} has invalid ttftMs`);
  assertEvidenceArray('benchmark', benchmark);
}

assert.ok(performanceSources.length >= 7, 'performance-sources.json needs broad methodology coverage');
for (const source of performanceSources) {
  assert.ok(source.id && typeof source.id === 'string', 'performance source needs id');
  assertEvidenceArray('performance source', { id: source.id, evidence: [source] });
}

const performanceSourceIds = new Set(performanceSources.map((source) => source.id));
for (const model of models) {
  for (const id of model.performance.methodEvidenceIds) {
    assert.ok(performanceSourceIds.has(id), `model ${model.slug} references missing performance source ${id}`);
  }
}

console.log(`data provenance ok: ${models.length} models, ${hardware.length} hardware profiles, ${benchmarks.length} benchmarks, ${performanceSources.length} performance sources`);
