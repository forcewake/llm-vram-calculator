const BYTES_IN_GB = 1024 ** 3;
const quantBytes = {
  fp32: 4,
  fp16: 2,
  nvfp6: 0.75,
  nvfp4: 0.5,
  q8: 8.5008 / 8,
  q7: 0.875,
  q6_k: 6.5633 / 8,
  q5_k_m: 5.7036 / 8,
  q5_k_s: 5.5704 / 8,
  q4_k_m: 4.8944 / 8,
  q4_k_s: 4.6672 / 8,
  q4: 0.5,
  q3_k_m: 3.9 / 8,
  q3_k_s: 3.5 / 8,
  q2_k: 2.9 / 8,
  fp8: 1,
  int4: 0.5
};

const activeParamsBySlug = {
  'deepseek-r1-671b': 37,
  'deepseek-v3': 37,
  'deepseek-v3-1': 37,
  'deepseek-v32': 37,
  'deepseek-v32-thinking': 37,
  'gpt-oss-20b': 3.6,
  'gpt-oss-120b': 5.1,
  'llama-4-scout': 17,
  'llama-4-maverick': 17,
  'minimax-text-01': 45.9,
  'minimax-vl-01': 45.9,
  'minimax-m1-40k': 45.9,
  'minimax-m1-80k': 45.9,
  'minimax-m2': 9.8,
  'minimax-m2-1': 9.8,
  'minimax-m2-5': 9.8,
  'minimax-m2-7': 10,
  'minimax-m3': 10,
  'mixtral-8x7b-v0-1': 12.9,
  'mixtral-8x22b-v0-1': 39
};

const runtimeProfiles = {
  generic: {
    label: 'Generic estimator',
    usableMemoryRatio: 0.94,
    speedMultiplier: 1,
    prefillMultiplier: 2.7,
    ttftMultiplier: 1,
    queueMultiplier: 1,
    overheadMultiplier: 1,
    lowFactor: 0.6,
    highFactor: 1.45,
    confidence: 'Directional'
  },
  mlx: {
    label: 'Apple MLX / Metal',
    usableMemoryRatio: 0.94,
    speedMultiplier: 1.65,
    prefillMultiplier: 2.4,
    ttftMultiplier: 0.92,
    queueMultiplier: 1.08,
    overheadMultiplier: 0.95,
    lowFactor: 0.5,
    highFactor: 1.25,
    confidence: 'Benchmark-calibrated'
  },
  llamacpp: {
    label: 'llama.cpp',
    usableMemoryRatio: 0.92,
    speedMultiplier: 1,
    prefillMultiplier: 2.3,
    ttftMultiplier: 1.08,
    queueMultiplier: 1.15,
    overheadMultiplier: 1.05,
    lowFactor: 0.55,
    highFactor: 1.35,
    confidence: 'Runtime-dependent'
  },
  ollama: {
    label: 'Ollama',
    usableMemoryRatio: 0.91,
    speedMultiplier: 0.88,
    prefillMultiplier: 2.2,
    ttftMultiplier: 1.14,
    queueMultiplier: 1.22,
    overheadMultiplier: 1.08,
    lowFactor: 0.55,
    highFactor: 1.25,
    confidence: 'Convenience runtime'
  },
  vllm: {
    label: 'vLLM',
    usableMemoryRatio: 0.9,
    speedMultiplier: 1.12,
    prefillMultiplier: 3.6,
    ttftMultiplier: 0.86,
    queueMultiplier: 0.72,
    overheadMultiplier: 1.12,
    lowFactor: 0.7,
    highFactor: 1.35,
    confidence: 'Serving-calibrated'
  },
  sglang: {
    label: 'SGLang',
    usableMemoryRatio: 0.9,
    speedMultiplier: 1.16,
    prefillMultiplier: 3.75,
    ttftMultiplier: 0.82,
    queueMultiplier: 0.7,
    overheadMultiplier: 1.12,
    lowFactor: 0.7,
    highFactor: 1.38,
    confidence: 'Serving-calibrated'
  },
  tensorrt: {
    label: 'TensorRT-LLM / NIM',
    usableMemoryRatio: 0.89,
    speedMultiplier: 1.28,
    prefillMultiplier: 4.1,
    ttftMultiplier: 0.76,
    queueMultiplier: 0.66,
    overheadMultiplier: 1.18,
    lowFactor: 0.75,
    highFactor: 1.42,
    confidence: 'Serving-calibrated'
  },
  mlc: {
    label: 'MLC-LLM',
    usableMemoryRatio: 0.92,
    speedMultiplier: 1.06,
    prefillMultiplier: 2.9,
    ttftMultiplier: 0.98,
    queueMultiplier: 1,
    overheadMultiplier: 1.04,
    lowFactor: 0.62,
    highFactor: 1.32,
    confidence: 'Runtime-dependent'
  },
  tgi: {
    label: 'Hugging Face TGI',
    usableMemoryRatio: 0.9,
    speedMultiplier: 1.08,
    prefillMultiplier: 3.35,
    ttftMultiplier: 0.9,
    queueMultiplier: 0.78,
    overheadMultiplier: 1.14,
    lowFactor: 0.68,
    highFactor: 1.32,
    confidence: 'Serving-calibrated'
  },
  lmdeploy: {
    label: 'LMDeploy',
    usableMemoryRatio: 0.9,
    speedMultiplier: 1.18,
    prefillMultiplier: 3.85,
    ttftMultiplier: 0.84,
    queueMultiplier: 0.72,
    overheadMultiplier: 1.13,
    lowFactor: 0.7,
    highFactor: 1.38,
    confidence: 'Serving-calibrated'
  },
  exllamav3: {
    label: 'ExLlamaV3',
    usableMemoryRatio: 0.91,
    speedMultiplier: 1.22,
    prefillMultiplier: 2.8,
    ttftMultiplier: 0.88,
    queueMultiplier: 0.92,
    overheadMultiplier: 1.07,
    lowFactor: 0.66,
    highFactor: 1.34,
    confidence: 'Runtime-dependent'
  }
};

export function calculateInference({ mode, model, hardware, weightQuant, kvQuant, trainingMethod, numGpus, batchSize, sequenceLength, concurrentUsers, offload, runtimeProfile = 'auto' }) {
  const resolvedRuntime = resolveRuntimeProfile(runtimeProfile, hardware);
  const runtime = runtimeProfiles[resolvedRuntime] || runtimeProfiles.generic;
  const availableGb = Math.max(1, hardware.memoryGb * numGpus * runtime.usableMemoryRatio);
  const isTraining = mode === 'finetune';
  const weightsGb = isTraining ? estimateFineTuneGb(model, trainingMethod, weightQuant) : estimateWeightsGb(model, weightQuant);
  const kvActiveGb = isTraining ? estimateKvCacheGb(model, 'fp16', sequenceLength, batchSize, 1) * 0.35 : estimateKvCacheGb(model, kvQuant, sequenceLength, batchSize, concurrentUsers);
  const kvCacheGb = offload && !isTraining ? kvActiveGb * 0.55 : kvActiveGb;
  const activationsGb = estimateActivationsGb(model, batchSize, sequenceLength) * (isTraining ? trainingActivationMultiplier(trainingMethod) : 1);
  const frameworkGb = estimateFrameworkOverheadGb(weightsGb, hardware.memoryGb, numGpus, runtime);
  const totalRequiredGb = weightsGb + kvCacheGb + activationsGb + frameworkGb;
  const utilization = (totalRequiredGb / availableGb) * 100;
  const headroomGb = availableGb - totalRequiredGb;
  const isRunnable = headroomGb >= 0;
  const pressurePenalty = utilization > 92 ? 0.45 : utilization > 82 ? 0.72 : utilization > 68 ? 0.88 : 1;
  const offloadPenalty = offload ? 0.68 : 1;
  const quantPenalty = getQuantPerformanceMultiplier(weightQuant);
  const baseBandwidth = hardware.bandwidthGbps * numGpus;
  const computeParams = getComputeParamsB(model);
  const modelScale = Math.max(1.1, Math.sqrt(computeParams.value));
  const decodeTokSec = Math.max(0.5, (baseBandwidth / (modelScale * 5.6)) * pressurePenalty * offloadPenalty * quantPenalty * runtime.speedMultiplier);
  const prefillTokSec = Math.max(0.5, (baseBandwidth / (modelScale * 2.25)) * pressurePenalty * offloadPenalty * quantPenalty * (runtime.prefillMultiplier || 2.7) * prefillContextPenalty(sequenceLength));
  const maxResidentSequences = estimateMaxResidentSequences({
    model,
    kvQuant,
    sequenceLength,
    batchSize,
    availableGb,
    weightsGb,
    activationsGb,
    frameworkGb,
    offload,
    isTraining
  });
  const queuedSequences = Math.max(0, concurrentUsers - maxResidentSequences);
  const interTokenLatencyMs = 1000 / Math.max(0.1, decodeTokSec);
  const queuePenaltyMs = Math.round(queuedSequences * interTokenLatencyMs * Math.min(256, Math.max(16, sequenceLength / 32)) * (runtime.queueMultiplier || 1));
  const ttftMs = Math.round((90 + (sequenceLength / Math.max(0.1, prefillTokSec)) * 1000 + Math.max(0, utilization - 75) * 8 + (offload ? 180 : 0)) * (runtime.ttftMultiplier || 1) + queuePenaltyMs);
  const totalThroughput = decodeTokSec * Math.min(concurrentUsers, Math.max(1, batchSize)) * (utilization > 98 ? 0.55 : 1);
  const tokensPerSecond = decodeTokSec;
  const msPerToken = interTokenLatencyMs;
  const timeToFirstTokenMs = ttftMs;
  const costPerHour = (hardware.priceHour || 0) * numGpus;
  const powerW = hardware.powerW * numGpus * (0.38 + Math.min(1, utilization / 100) * 0.62);
  const co2KgDay = (powerW / 1000) * 24 * 0.37;
  const tokensPerStep = batchSize * sequenceLength;
  const stepsPerHour = Math.max(0.1, (baseBandwidth / (modelScale * 3.8)) * pressurePenalty * (trainingMethod === 'full' ? 0.34 : trainingMethod === 'qlora' ? 0.74 : 0.86));
  const trainingTokensHour = tokensPerStep * stepsPerHour;

  const allocation = [
    { key: 'weights', label: isTraining ? 'Trainable State' : 'Base Model Weights', gb: weightsGb, color: '#2f9bf4' },
    { key: 'activations', label: 'Activations', gb: activationsGb, color: '#20c997' },
    { key: 'kv', label: isTraining ? 'Attention Cache' : offload ? 'Resident KV Cache' : 'KV Cache', gb: kvCacheGb, color: '#c753e8' },
    { key: 'framework', label: 'Runtime Overhead', gb: frameworkGb, color: '#f3b312' }
  ];

  return {
    availableGb,
    weightsGb,
    kvCacheGb,
    activationsGb,
    frameworkGb,
    totalRequiredGb,
    utilization,
    headroomGb,
    overBudgetGb: Math.max(0, -headroomGb),
    isRunnable,
    decodeTokSec,
    prefillTokSec,
    ttftMs,
    interTokenLatencyMs,
    maxResidentSequences,
    queuePenaltyMs,
    tokensPerSecond,
    totalThroughput,
    tokensPerStep,
    stepsPerHour,
    trainingTokensHour,
    msPerToken,
    timeToFirstTokenMs,
    speedLow: decodeTokSec * runtime.lowFactor,
    speedHigh: decodeTokSec * runtime.highFactor,
    computeParamsB: computeParams.value,
    computeParamSource: computeParams.source,
    totalParamsB: model.paramsB || computeParams.value,
    weightBpw: getWeightBitsPerParam(model, weightQuant),
    kvGbPer1kTokens: estimateKvCacheGb(model, kvQuant, 1024, 1, 1),
    runtimeProfile: resolvedRuntime,
    runtimeLabel: runtime.label,
    estimateConfidence: runtime.confidence,
    costPerHour,
    powerW,
    co2KgDay,
    allocation
  };
}

function estimateWeightsGb(model, weightQuant) {
  const direct = model.vram?.[weightQuant];
  if (direct) return direct;
  if (weightQuant.startsWith('q4') && model.vram?.q4) return model.vram.q4 * q4VariantMultiplier(weightQuant);
  if (weightQuant === 'q4' && model.vram?.q4) return model.vram.q4;
  if (weightQuant === 'q8' && model.vram?.q8) return model.vram.q8;
  if (model.vram?.fp16) {
    return model.vram.fp16 * ((quantBytes[weightQuant] || 2) / quantBytes.fp16) * quantOverhead(weightQuant);
  }
  const params = model.paramsB || 7;
  const bytes = quantBytes[weightQuant] || 2;
  return params * 1e9 * bytes / BYTES_IN_GB * 1.08 * quantOverhead(weightQuant);
}

function prefillContextPenalty(sequenceLength) {
  if (sequenceLength <= 4096) return 1;
  if (sequenceLength <= 16384) return 0.9;
  if (sequenceLength <= 65536) return 0.78;
  return 0.64;
}

function estimateMaxResidentSequences({ model, kvQuant, sequenceLength, batchSize, availableGb, weightsGb, activationsGb, frameworkGb, offload, isTraining }) {
  if (isTraining) return Math.max(1, batchSize);
  const residentBudgetGb = availableGb - weightsGb - activationsGb - frameworkGb;
  if (residentBudgetGb <= 0) return 0;
  const kvGbPerSequence = estimateKvCacheGb(model, kvQuant, sequenceLength, batchSize, 1) * (offload ? 0.55 : 1);
  if (kvGbPerSequence <= 0) return 0;
  return Math.max(0, Math.floor(residentBudgetGb / kvGbPerSequence));
}

function q4VariantMultiplier(weightQuant) {
  if (weightQuant === 'q4_k_m') return 1.04;
  if (weightQuant === 'q4_k_s') return 1.01;
  return 1;
}

function getComputeParamsB(model) {
  if (model.activeParamsB) return { value: model.activeParamsB, source: 'model metadata' };
  if (activeParamsBySlug[model.slug]) return { value: activeParamsBySlug[model.slug], source: 'known MoE active params' };
  const activeMatch = `${model.name || ''} ${model.slug || ''}`.match(/a(\d+(?:\.\d+)?)b/i);
  if (model.architecture === 'moe' && activeMatch) return { value: Number(activeMatch[1]), source: 'parsed from model name' };
  return { value: model.paramsB || 7, source: model.architecture === 'moe' ? 'total params fallback' : 'dense model params' };
}

function resolveRuntimeProfile(runtimeProfile, hardware) {
  if (runtimeProfile && runtimeProfile !== 'auto') return runtimeProfile;
  if (hardware.type === 'Unified Memory' && hardware.vendor === 'Apple') return 'mlx';
  if (hardware.type === 'Datacenter GPU') return 'vllm';
  return 'llamacpp';
}

function getWeightBitsPerParam(model, weightQuant) {
  const directGb = estimateWeightsGb(model, weightQuant);
  const params = model.paramsB || 7;
  if (directGb && params) return (directGb * BYTES_IN_GB * 8) / (params * 1e9);
  return (quantBytes[weightQuant] || 2) * 8;
}

function quantOverhead(weightQuant) {
  if (weightQuant === 'fp32' || weightQuant === 'fp16') return 1;
  if (weightQuant === 'nvfp6' || weightQuant === 'nvfp4') return 1.04;
  if (weightQuant.includes('_k')) return 1.08;
  return 1.06;
}

function getQuantPerformanceMultiplier(weightQuant) {
  const multipliers = {
    fp32: 0.82,
    fp16: 1,
    nvfp6: 0.98,
    nvfp4: 0.94,
    q8: 0.96,
    q7: 0.95,
    q6_k: 0.94,
    q5_k_m: 0.92,
    q5_k_s: 0.91,
    q4_k_m: 0.9,
    q4_k_s: 0.88,
    q4: 0.89,
    q3_k_m: 0.84,
    q3_k_s: 0.82,
    q2_k: 0.76
  };
  return multipliers[weightQuant] || 1;
}

function estimateFineTuneGb(model, trainingMethod, weightQuant) {
  const direct = {
    full: model.vram?.finetuneFull,
    lora: model.vram?.finetuneLora,
    qlora: model.vram?.finetuneQlora
  }[trainingMethod];
  if (direct) return direct;
  const base = estimateWeightsGb(model, weightQuant);
  if (trainingMethod === 'full') return base * 6.2;
  if (trainingMethod === 'qlora') return base * 1.55 + 1.1;
  return base * 2.1 + 0.8;
}

function trainingActivationMultiplier(trainingMethod) {
  if (trainingMethod === 'full') return 4.4;
  if (trainingMethod === 'qlora') return 2.0;
  return 2.35;
}

function estimateKvCacheGb(model, kvQuant, sequenceLength, batchSize, concurrentUsers) {
  const layers = model.layers || 32;
  const effectiveSequence = model.slidingWindow ? Math.min(sequenceLength, model.slidingWindow) : sequenceLength;
  const bytes = quantBytes[kvQuant] || 2;
  if (model.attentionStructure === 'mla') {
    const mlaCacheDim = (model.kvLoraRank || 512) + (model.qkRopeHeadDim || 64);
    const cacheBytes = layers * mlaCacheDim * effectiveSequence * batchSize * concurrentUsers * bytes;
    return cacheBytes / BYTES_IN_GB * 1.08;
  }
  const kvHeads = model.kvHeads || model.attentionHeads || 32;
  const headDim = model.headDim || (model.hiddenSize && model.attentionHeads ? model.hiddenSize / model.attentionHeads : 128);
  const cacheBytes = 2 * layers * kvHeads * headDim * effectiveSequence * batchSize * concurrentUsers * bytes;
  return cacheBytes / BYTES_IN_GB * 1.08;
}

function estimateActivationsGb(model, batchSize, sequenceLength) {
  const hidden = model.hiddenSize || 4096;
  const layers = model.layers || 32;
  const scaledTokens = Math.sqrt(Math.max(1, sequenceLength / 1024));
  return Math.max(0.18, (hidden * layers * batchSize * scaledTokens * 2) / BYTES_IN_GB * 14);
}

function estimateFrameworkOverheadGb(weightsGb, memoryGb, numGpus, runtime) {
  const base = Math.max(0.65 * numGpus, Math.min(memoryGb * numGpus * 0.12, weightsGb * 0.12 + 0.34 * numGpus));
  return base * (runtime?.overheadMultiplier || 1);
}

export function getFitStatus(utilization) {
  if (utilization <= 70) return { label: 'Healthy', tone: 'good' };
  if (utilization <= 88) return { label: 'Moderate', tone: 'ok' };
  if (utilization <= 100) return { label: 'Tight', tone: 'warn' };
  return { label: 'Over Budget', tone: 'bad' };
}

export function formatGb(value) {
  if (!Number.isFinite(value)) return '0 GB';
  if (value >= 100) return `${Math.round(value)} GB`;
  if (value >= 10) return `${value.toFixed(1)} GB`;
  return `${value.toFixed(2)} GB`;
}

export function formatNumber(value) {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: value < 10 ? 1 : 0 }).format(value);
}
