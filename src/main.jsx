import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ActionIcon,
  Button,
  MantineProvider,
  NumberInput,
  Select,
  SegmentedControl,
  Slider,
  Switch,
  Tooltip
} from '@mantine/core';
import '@mantine/core/styles.css';
import {
  Activity,
  Brain,
  Copy,
  Cpu,
  Database,
  ExternalLink,
  Gauge,
  Info,
  Plus,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Zap
} from 'lucide-react';
import models from './data/models.json';
import hardwareProfiles from './data/hardware.json';
import observedBenchmarks from './data/benchmarks.json';
import performanceSources from './data/performance-sources.json';
import { calculateInference, formatGb, formatNumber, getFitStatus } from './lib/calculator';
import './styles.css';

const weightOptions = [
  { id: 'fp32', label: 'FP32', bits: 32, quality: 'Maximum', score: 100, description: 'Full precision weights. Mostly useful as a baseline.' },
  { id: 'fp16', label: 'FP16 / BF16', bits: 16, quality: 'Maximum', score: 96, description: 'Default high-compatibility inference precision.' },
  { id: 'q8', label: 'Q8', bits: 8, quality: 'Very high', score: 92, description: '8-bit quantized weights with strong quality retention.' },
  { id: 'q7', label: 'Q7', bits: 7, quality: 'Very high', score: 90, description: '7-bit quantized estimate between Q8 and Q6.' },
  { id: 'nvfp6', label: 'NVFP6', bits: 6, quality: 'High', score: 88, description: '6-bit floating-point style estimate for newer NVIDIA stacks.' },
  { id: 'q6_k', label: 'Q6_K', bits: 6, quality: 'High', score: 86, description: 'GGUF-style K quantization with higher quality than Q5/Q4.' },
  { id: 'q5_k_m', label: 'Q5_K_M', bits: 5, quality: 'High', score: 82, description: 'Medium 5-bit K quantization.' },
  { id: 'q5_k_s', label: 'Q5_K_S', bits: 5, quality: 'Medium high', score: 80, description: 'Smaller 5-bit K quantization.' },
  { id: 'nvfp4', label: 'NVFP4', bits: 4, quality: 'Medium', score: 78, description: '4-bit floating-point style estimate for newer NVIDIA stacks.' },
  { id: 'q4_k_m', label: 'Q4_K_M', bits: 4, quality: 'Medium', score: 76, description: 'Common balanced 4-bit K quantization.' },
  { id: 'q4_k_s', label: 'Q4_K_S', bits: 4, quality: 'Medium low', score: 73, description: 'Smaller 4-bit K quantization.' },
  { id: 'q4', label: 'Q4', bits: 4, quality: 'Medium low', score: 72, description: 'Generic 4-bit quantized estimate.' },
  { id: 'q3_k_m', label: 'Q3_K_M', bits: 3, quality: 'Low', score: 64, description: 'Aggressive 3-bit K quantization.' },
  { id: 'q3_k_s', label: 'Q3_K_S', bits: 3, quality: 'Low', score: 60, description: 'Smaller aggressive 3-bit K quantization.' },
  { id: 'q2_k', label: 'Q2_K', bits: 2, quality: 'Very low', score: 48, description: 'Very aggressive 2-bit K quantization; quality risk is high.' }
];

const kvOptions = [
  { id: 'fp16', label: 'FP16 / BF16', bits: 16, description: 'Default high-precision cache' },
  { id: 'fp8', label: 'FP8 KV cache', bits: 8, description: 'Modern serving stacks can halve cache memory' },
  { id: 'int4', label: 'INT4 KV cache', bits: 4, description: 'Aggressive long-context compression' }
];

const trainingOptions = [
  { id: 'lora', label: 'LoRA', description: 'Adapter fine-tuning with modest optimizer memory' },
  { id: 'qlora', label: 'QLoRA', description: 'Quantized base model with adapter training' },
  { id: 'full', label: 'Full SFT', description: 'Updates all weights and needs the most VRAM' }
];

const runtimeOptions = [
  { value: 'auto', label: 'Auto runtime profile' },
  { value: 'mlx', label: 'Apple MLX / Metal' },
  { value: 'llamacpp', label: 'llama.cpp' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'vllm', label: 'vLLM / TGI serving' },
  { value: 'sglang', label: 'SGLang serving' },
  { value: 'tensorrt', label: 'TensorRT-LLM / NIM' },
  { value: 'mlc', label: 'MLC-LLM' },
  { value: 'tgi', label: 'TGI' },
  { value: 'lmdeploy', label: 'LMDeploy' },
  { value: 'exllamav3', label: 'ExLlamaV3' },
  { value: 'generic', label: 'Generic estimator' }
];

const defaultModel = models.find((model) => model.slug === 'deepseek-r1-3b') || models[0];
const defaultHardware = hardwareProfiles.find((gpu) => gpu.id === 'rtx3060') || hardwareProfiles[0];
const defaultCompareModels = ['deepseek-r1-3b', 'llama-3-1-8b', 'qwen3-8b']
  .map((slug) => models.find((model) => model.slug === slug))
  .filter(Boolean);

function formatModelOption(model) {
  return `${model.provider} ${model.name}${model.paramsB ? ` (${model.paramsB}B)` : ''}`;
}

function createProfileId(prefix = 'profile') {
  return globalThis.crypto?.randomUUID?.() || `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isSparseModel(model) {
  return model.architecture === 'moe';
}

function getModelParams(model) {
  if (model.paramsB === null || model.paramsB === undefined || model.paramsB === '') {
    return Number.POSITIVE_INFINITY;
  }

  const paramsB = Number(model.paramsB);
  return Number.isFinite(paramsB) ? paramsB : Number.POSITIVE_INFINITY;
}

function compareModelParams(a, b) {
  const aParams = getModelParams(a);
  const bParams = getModelParams(b);
  if (aParams === bParams) return 0;
  return aParams < bParams ? -1 : 1;
}

function compareModelsForSelect(a, b) {
  const architectureDelta = Number(isSparseModel(a)) - Number(isSparseModel(b));
  if (architectureDelta !== 0) return architectureDelta;

  const paramsDelta = compareModelParams(a, b);
  if (paramsDelta !== 0) return paramsDelta;

  return formatModelOption(a).localeCompare(formatModelOption(b));
}

function compareModelFamilies([familyA, modelsA], [familyB, modelsB]) {
  const aHasDense = modelsA.some((model) => !isSparseModel(model));
  const bHasDense = modelsB.some((model) => !isSparseModel(model));
  if (aHasDense !== bHasDense) return aHasDense ? -1 : 1;

  const aReference = modelsA.filter((model) => !isSparseModel(model)).sort(compareModelsForSelect)[0] || modelsA.sort(compareModelsForSelect)[0];
  const bReference = modelsB.filter((model) => !isSparseModel(model)).sort(compareModelsForSelect)[0] || modelsB.sort(compareModelsForSelect)[0];
  const paramsDelta = compareModelParams(aReference, bReference);
  if (paramsDelta !== 0) return paramsDelta;

  return familyA.localeCompare(familyB);
}

function getModelSearchText(model) {
  return [
    formatModelOption(model),
    model.name,
    model.family,
    model.provider,
    model.slug,
    model.architecture,
    model.attentionStructure,
    model.paramsB ? `${model.paramsB}B` : ''
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

function groupModelOptions(modelList) {
  const groups = modelList.reduce((acc, model) => {
    const group = model.family || 'Other models';
    acc[group] = acc[group] || [];
    acc[group].push(model);
    return acc;
  }, {});

  return Object.entries(groups)
    .sort(compareModelFamilies)
    .map(([group, groupModels]) => ({
      group,
      items: groupModels.sort(compareModelsForSelect).map((model) => ({
        value: String(model.id),
        label: formatModelOption(model)
      }))
    }));
}

const contextPresets = [
  { value: 512, label: '512 - Smoke test' },
  { value: 1024, label: '1K - Short chat' },
  { value: 2048, label: '2K - Prompt eval' },
  { value: 4096, label: '4K - Chat default' },
  { value: 8192, label: '8K - Long chat' },
  { value: 16384, label: '16K - Docs' },
  { value: 32768, label: '32K - RAG / reports' },
  { value: 65536, label: '64K - Large repo' },
  { value: 131072, label: '128K - Book scale' },
  { value: 262144, label: '256K - Corpus' },
  { value: 1048576, label: '1M - Extreme' }
];

function formatContextLabel(tokens) {
  if (tokens >= 1048576) return `${formatNumber(tokens / 1048576)}M tokens`;
  if (tokens >= 1024) return `${formatNumber(tokens / 1024)}K tokens`;
  return `${formatNumber(tokens)} tokens`;
}

function contextOptionsForModel(maxContext, currentContext) {
  const safeMax = Math.max(512, maxContext || 32768);
  const safeCurrent = clamp(Number(currentContext), 512, safeMax);
  const optionMap = new Map();

  contextPresets
    .filter((option) => option.value <= safeMax)
    .forEach((option) => {
      optionMap.set(option.value, {
        value: String(option.value),
        label: option.label
      });
    });

  if (!optionMap.has(safeCurrent)) {
    optionMap.set(safeCurrent, {
      value: String(safeCurrent),
      label: `${formatContextLabel(safeCurrent)} - Current`
    });
  }

  if (!optionMap.has(safeMax)) {
    optionMap.set(safeMax, {
      value: String(safeMax),
      label: `${formatContextLabel(safeMax)} - Max`
    });
  }

  return [...optionMap.entries()]
    .sort(([a], [b]) => a - b)
    .map(([, option]) => option);
}

function getQuantizationRows({ mode, model, hardware, kvQuant, trainingMethod, numGpus, batchSize, sequenceLength, concurrentUsers, offload, runtimeProfile }) {
  return weightOptions.map((option) => {
    const result = calculateInference({
      mode,
      model,
      hardware,
      weightQuant: option.id,
      kvQuant,
      trainingMethod,
      numGpus,
      batchSize,
      sequenceLength,
      concurrentUsers,
      offload,
      runtimeProfile
    });
    const fit = getFitStatus(result.utilization);
    const grade = getFitGrade(result.utilization);
    return { option, result, fit, grade };
  });
}

function getWeightSelectOptions(rows) {
  return rows.map(({ option, result, fit }) => ({
    value: option.id,
    label: option.label,
    bits: option.bits,
    quality: option.quality,
    score: option.score,
    description: option.description,
    memoryLabel: formatGb(result.totalRequiredGb),
    memoryPercent: Math.round(result.utilization),
    fitLabel: fit.label,
    fitTone: fit.tone
  }));
}

function isRunnableResult(result) {
  return result.isRunnable ?? result.headroomGb >= 0;
}

function formatRunRate(mode, result) {
  if (!isRunnableResult(result)) return 'Not runnable';
  return mode === 'inference'
    ? `~${formatNumber(getDecodeTokSec(result))} tok/sec`
    : `~${formatNumber(result.trainingTokensHour)}`;
}

function overBudgetHint(result) {
  return `Needs ${formatGb(result.overBudgetGb || Math.abs(result.headroomGb))} more usable memory. Lower precision, reduce context/batch, or add GPUs.`;
}

function getDecodeTokSec(result) {
  return result.decodeTokSec ?? result.tokensPerSecond ?? 0;
}

function getPrefillTokSec(result) {
  return result.prefillTokSec ?? Math.max(getDecodeTokSec(result) * 8, getDecodeTokSec(result));
}

function getTtftMs(result) {
  return result.ttftMs ?? result.timeToFirstTokenMs ?? 0;
}

function formatDeltaPercent(observed, estimated) {
  if (!Number.isFinite(observed) || !Number.isFinite(estimated) || estimated <= 0) return null;
  const delta = ((observed - estimated) / estimated) * 100;
  const sign = delta >= 0 ? '+' : '';
  return `${sign}${Math.round(delta)}% vs est.`;
}

function nearestContextBenchmarks(benchmarks, contextTokens, limit = 8) {
  return benchmarks
    .slice()
    .sort((a, b) => Math.abs((a.contextTokens || contextTokens) - contextTokens) - Math.abs((b.contextTokens || contextTokens) - contextTokens))
    .slice(0, limit);
}

function hardwareFamily(id = '') {
  if (id.startsWith('m5max-')) return 'm5max';
  if (id.startsWith('m4max-')) return 'm4max';
  return id;
}

function getMatchingBenchmarks(model, hardware, numGpus) {
  return observedBenchmarks
    .filter((benchmark) => benchmark.modelSlug === model.slug)
    .filter((benchmark) => {
      const exactHardware = benchmark.hardwareId === hardware.id;
      const sameFamily = benchmark.hardwareId && hardware.id && hardwareFamily(benchmark.hardwareId) === hardwareFamily(hardware.id);
      return exactHardware || sameFamily;
    })
    .filter((benchmark) => !benchmark.numGpus || benchmark.numGpus === numGpus)
    .sort((a, b) => {
      const contextDelta = (a.contextTokens || 0) - (b.contextTokens || 0);
      if (contextDelta !== 0) return contextDelta;
      return `${a.engineLabel} ${a.quantization}`.localeCompare(`${b.engineLabel} ${b.quantization}`);
    });
}

function getFitRecommendations({ result, quantizationRows, currentQuant, sequenceLength, batchSize, concurrentUsers, numGpus, offload, hardware, hardwareProfiles, selectedModel }) {
  if (isRunnableResult(result)) return [];

  const recommendations = [];
  const fittingQuant = quantizationRows
    .filter((row) => row.result.isRunnable && row.option.id !== currentQuant)
    .sort((a, b) => b.option.score - a.option.score || a.result.totalRequiredGb - b.result.totalRequiredGb)[0];

  if (fittingQuant) {
    recommendations.push({
      id: 'quant',
      title: `Switch weights to ${fittingQuant.option.label}`,
      body: `${formatGb(fittingQuant.result.totalRequiredGb)} required, ${Math.round(fittingQuant.result.utilization)}% of usable memory. Quality score ${fittingQuant.option.score}.`,
      actionLabel: `Use ${fittingQuant.option.label}`,
      action: { type: 'quant', value: fittingQuant.option.id }
    });
  }

  const targetContext = [32768, 16384, 8192, 4096, 2048, 1024, 512].find((value) => value < sequenceLength);
  if (targetContext) {
    recommendations.push({
      id: 'context',
      title: `Reduce context to ${formatContextLabel(targetContext)}`,
      body: 'Cuts KV cache and prefill latency. This helps most for long-context workloads.',
      actionLabel: `Use ${formatContextLabel(targetContext)}`,
      action: { type: 'context', value: targetContext }
    });
  }

  if (!offload) {
    recommendations.push({
      id: 'offload',
      title: 'Enable CPU/RAM or NVMe offload',
      body: 'Can reduce resident KV pressure, but expect lower speed and higher latency.',
      actionLabel: 'Enable offload',
      action: { type: 'offload' }
    });
  }

  const neededGpus = Math.ceil(result.totalRequiredGb / Math.max(1, result.availableGb / numGpus));
  if (neededGpus > numGpus && neededGpus <= 16) {
    recommendations.push({
      id: 'gpus',
      title: `Increase to ${neededGpus} GPUs`,
      body: `Current run needs ${formatGb(result.totalRequiredGb)}; ${neededGpus} similar devices should provide enough usable memory.`,
      actionLabel: `Use ${neededGpus} GPUs`,
      action: { type: 'gpus', value: neededGpus }
    });
  }

  const fittingHardware = hardwareProfiles
    .filter((candidate) => candidate.id !== hardware.id)
    .map((candidate) => {
      const runtimeUsableRatio = result.availableGb / Math.max(1, hardware.memoryGb * numGpus);
      const usableGb = candidate.memoryGb * numGpus * runtimeUsableRatio;
      return { candidate, usableGb };
    })
    .filter(({ usableGb }) => usableGb >= result.totalRequiredGb)
    .sort((a, b) => a.usableGb - b.usableGb)[0];

  if (fittingHardware) {
    recommendations.push({
      id: 'hardware',
      title: `Switch to ${fittingHardware.candidate.vendor} ${fittingHardware.candidate.name}`,
      body: `${formatGb(fittingHardware.usableGb)} approximate usable memory with ${numGpus} device${numGpus > 1 ? 's' : ''}.`,
      actionLabel: `Use ${fittingHardware.candidate.name}`,
      action: { type: 'hardware', value: fittingHardware.candidate.id }
    });
  }

  if (!recommendations.length) {
    recommendations.push({
      id: 'none',
      title: 'No single quick fix fits this run',
      body: `${selectedModel.name} needs ${formatGb(result.totalRequiredGb)} with this shape. Combine lower precision, shorter context, and more/larger devices.`,
      actionLabel: null
    });
  }

  return recommendations.slice(0, 5);
}

function getQualityTone(score) {
  if (score >= 88) return 'good';
  if (score >= 72) return 'ok';
  if (score >= 60) return 'warn';
  return 'bad';
}

function getFitGrade(utilization) {
  if (utilization <= 70) return { grade: 'S', score: clamp(Math.round(98 - utilization * 0.08), 90, 98), tone: 'good' };
  if (utilization <= 88) return { grade: 'A', score: clamp(Math.round(92 - (utilization - 70) * 0.45), 84, 91), tone: 'ok' };
  if (utilization <= 100) return { grade: 'B', score: clamp(Math.round(83 - (utilization - 88) * 0.7), 74, 83), tone: 'warn' };
  return { grade: 'C', score: clamp(Math.round(72 - (utilization - 100) * 0.7), 20, 72), tone: 'bad' };
}

function renderWeightOption({ option, checked }) {
  const quant = weightOptions.find((item) => item.id === option.value);
  if (!quant) return option.label;
  const memoryLabel = option.memoryLabel;
  const memoryPercent = option.memoryPercent;
  const fitLabel = option.fitLabel;
  const fitTone = option.fitTone || getQualityTone(quant.score);

  return (
    <div className="rich-select-option">
      <div>
        <strong>{quant.label}</strong>
        <span>{quant.bits}-bit · {quant.quality}</span>
        <small>{quant.description}</small>
      </div>
      <div className="rich-option-metrics">
        <em className={`mini-badge ${fitTone}`}>{memoryLabel || (checked ? 'Selected' : quant.score)}</em>
        {memoryLabel && <small>{memoryPercent}% · {fitLabel}</small>}
      </div>
    </div>
  );
}

function createProfile(seed = {}) {
  const model = seed.model || defaultCompareModels[0] || defaultModel;
  return {
    id: seed.id || createProfileId(),
    modelId: String(seed.modelId || model.id),
    hardwareId: seed.hardwareId || defaultHardware.id,
    weightQuant: seed.weightQuant || 'fp16',
    kvQuant: seed.kvQuant || 'fp16',
    trainingMethod: seed.trainingMethod || 'lora',
    numGpus: seed.numGpus || 1,
    sequenceLength: seed.sequenceLength || 1024
  };
}

function App() {
  const [mode, setMode] = useState('inference');
  const [selectedModelId, setSelectedModelId] = useState(defaultModel.id);
  const [selectedHardwareId, setSelectedHardwareId] = useState(defaultHardware.id);
  const [customVram, setCustomVram] = useState('');
  const [weightQuant, setWeightQuant] = useState('fp16');
  const [kvQuant, setKvQuant] = useState('fp16');
  const [trainingMethod, setTrainingMethod] = useState('lora');
  const [numGpus, setNumGpus] = useState(1);
  const [batchSize, setBatchSize] = useState(1);
  const [sequenceLength, setSequenceLength] = useState(1024);
  const [concurrentUsers, setConcurrentUsers] = useState(1);
  const [offload, setOffload] = useState(false);
  const [runtimeProfile, setRuntimeProfile] = useState('auto');
  const [query, setQuery] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showQuantizationMatrix, setShowQuantizationMatrix] = useState(false);
  const [compareProfiles, setCompareProfiles] = useState(() =>
    (defaultCompareModels.length ? defaultCompareModels : [defaultModel]).map((model, index) =>
      createProfile({
        id: `profile-${model.id}-${index}`,
        model,
        weightQuant: index === 0 ? 'fp16' : index === 1 ? 'q4_k_m' : 'q8'
      })
    )
  );

  const selectedModel = models.find((model) => model.id === Number(selectedModelId)) || defaultModel;
  const selectedHardware = hardwareProfiles.find((gpu) => gpu.id === selectedHardwareId) || defaultHardware;
  const hardware = {
    ...selectedHardware,
    memoryGb: customVram ? Number(customVram) : selectedHardware.memoryGb
  };
  const effectiveSequenceLength = Math.min(sequenceLength, selectedModel.contextLength || 32768);

  const result = useMemo(
    () =>
      calculateInference({
        mode,
        model: selectedModel,
        hardware,
        weightQuant,
        kvQuant,
        trainingMethod,
        numGpus,
        batchSize,
        sequenceLength: effectiveSequenceLength,
        concurrentUsers,
        offload,
        runtimeProfile
      }),
    [mode, selectedModel, hardware, weightQuant, kvQuant, trainingMethod, numGpus, batchSize, effectiveSequenceLength, concurrentUsers, offload, runtimeProfile]
  );
  const quantizationRows = useMemo(
    () =>
      getQuantizationRows({
        mode,
        model: selectedModel,
        hardware,
        kvQuant,
        trainingMethod,
        numGpus,
        batchSize,
        sequenceLength: effectiveSequenceLength,
        concurrentUsers,
        offload,
        runtimeProfile
      }),
    [mode, selectedModel, hardware, kvQuant, trainingMethod, numGpus, batchSize, effectiveSequenceLength, concurrentUsers, offload, runtimeProfile]
  );
  const weightSelectOptions = useMemo(() => getWeightSelectOptions(quantizationRows), [quantizationRows]);

  const fit = getFitStatus(result.utilization);
  const canRun = isRunnableResult(result);
  const matchingBenchmarks = useMemo(
    () => getMatchingBenchmarks(selectedModel, selectedHardware, numGpus),
    [selectedModel, selectedHardware, numGpus]
  );
  const fitRecommendations = useMemo(
    () =>
      getFitRecommendations({
        result,
        quantizationRows,
        currentQuant: weightQuant,
        sequenceLength: effectiveSequenceLength,
        numGpus,
        offload,
        hardware,
        hardwareProfiles,
        selectedModel
      }),
    [result, quantizationRows, weightQuant, effectiveSequenceLength, numGpus, offload, hardware, selectedModel]
  );
  const modelOptions = useMemo(() => groupModelOptions(models), []);
  const filteredModels = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) return modelOptions;
    return groupModelOptions(models.filter((model) => getModelSearchText(model).includes(term)));
  }, [modelOptions, query]);
  const hardwareOptions = useMemo(() => {
    const groups = hardwareProfiles.reduce((acc, gpu) => {
      const group = gpu.type || 'Other Hardware';
      acc[group] = acc[group] || [];
      acc[group].push({
        value: gpu.id,
        label: `${gpu.vendor} ${gpu.name} (${gpu.memoryGb}GB)`
      });
      return acc;
    }, {});

    return Object.entries(groups).map(([group, items]) => ({ group, items }));
  }, []);
  const updateCompareProfile = (profileId, patch) => {
    setCompareProfiles((currentProfiles) =>
      currentProfiles.map((profile) => (profile.id === profileId ? { ...profile, ...patch } : profile))
    );
  };
  const addCompareProfile = () => {
    setCompareProfiles((currentProfiles) => [
      ...currentProfiles,
      createProfile({
        id: `profile-${Date.now()}`,
        modelId: String(selectedModel.id),
        hardwareId: selectedHardware.id,
        weightQuant,
        kvQuant,
        trainingMethod,
        numGpus,
        sequenceLength: effectiveSequenceLength
      })
    ]);
  };
  const cloneCompareProfile = (profileId) => {
    setCompareProfiles((currentProfiles) => {
      if (currentProfiles.length >= 6) return currentProfiles;

      const sourceIndex = currentProfiles.findIndex((profile) => profile.id === profileId);
      if (sourceIndex === -1) return currentProfiles;

      const { id, ...settings } = currentProfiles[sourceIndex];
      const clonedProfile = {
        ...settings,
        id: createProfileId('profile-copy')
      };

      return [
        ...currentProfiles.slice(0, sourceIndex + 1),
        clonedProfile,
        ...currentProfiles.slice(sourceIndex + 1)
      ];
    });
  };
  const removeCompareProfile = (profileId) => {
    setCompareProfiles((currentProfiles) => currentProfiles.filter((profile) => profile.id !== profileId));
  };
  const applyRecommendation = (action) => {
    if (!action) return;
    if (action.type === 'quant') setWeightQuant(action.value);
    if (action.type === 'context') setSequenceLength(action.value);
    if (action.type === 'offload') setOffload(true);
    if (action.type === 'gpus') setNumGpus(action.value);
    if (action.type === 'hardware') {
      setSelectedHardwareId(action.value);
      setCustomVram('');
    }
  };

  return (
    <main>
      <div className="app-shell">
        <section className="calculator-paper">
          <header className="topbar">
            <div>
              <h1>{mode === 'inference' ? 'LLM Inference: VRAM & Performance Calculator' : `${trainingMethod.toUpperCase()} Fine-tuning: VRAM & Cost Calculator`}</h1>
            </div>
          </header>

          <SegmentedControl
            className="mode-tabs mantine-mode-tabs"
            value={mode}
            onChange={setMode}
            data={[
              { value: 'inference', label: <span className="segment-label"><Zap size={18} />Inference</span> },
              { value: 'finetune', label: <span className="segment-label"><Brain size={18} />Fine-tuning</span> }
            ]}
            size="md"
            radius="md"
          />

          <div className="layout-grid">
            <section className="control-panel" aria-label="Calculator controls">
              <div className="panel-heading">
                <div>
                  <h2>Configure Run</h2>
                  <p>Pick a model and serving shape. Every control updates the memory ledger instantly.</p>
                </div>
                <Tooltip label="Advanced controls" withArrow>
                <ActionIcon className="icon-button" variant="light" color="gray" size="xl" radius="md" type="button" onClick={() => setShowAdvanced((value) => !value)} aria-label="Toggle advanced details">
                  <SlidersHorizontal size={18} />
                </ActionIcon>
                </Tooltip>
              </div>

              <Select
                className="field-group mantine-field"
                label="Select Model"
                required
                searchable
                searchValue={query}
                onSearchChange={setQuery}
                onDropdownOpen={() => setQuery('')}
                onDropdownClose={() => setQuery('')}
                data={filteredModels}
                value={String(selectedModelId)}
                onChange={(value) => {
                  if (value) {
                    setSelectedModelId(value);
                    setQuery('');
                  }
                }}
                leftSection={<Search size={17} />}
                placeholder="Search model, family, provider..."
                nothingFoundMessage="No model found"
                maxDropdownHeight={320}
                comboboxProps={{ shadow: 'lg' }}
              />

              <div className="two-columns">
                {mode === 'inference' ? (
                  <>
                    <SelectField label="Inference Quantization" value={weightQuant} onChange={setWeightQuant} options={weightOptions} selectData={weightSelectOptions} renderOption={renderWeightOption} leftSection={<Gauge size={17} />} />
                    <SegmentedField label="KV Cache Quantization" value={kvQuant} onChange={setKvQuant} options={kvOptions} />
                  </>
                ) : (
                  <>
                    <SegmentedField label="Training Strategy" value={trainingMethod} onChange={setTrainingMethod} options={trainingOptions} />
                    <SelectField label="Checkpoint Precision" value={weightQuant} onChange={setWeightQuant} options={weightOptions} selectData={weightSelectOptions} renderOption={renderWeightOption} leftSection={<Gauge size={17} />} />
                  </>
                )}
              </div>

              <div className="two-columns hardware-row">
                <label className="field-group">
                  <span>Hardware Configuration</span>
                  <small>Select your GPU or set custom VRAM.</small>
                  <Select
                    data={hardwareOptions}
                    value={selectedHardwareId}
                    onChange={(value) => value && setSelectedHardwareId(value)}
                    searchable
                    nothingFoundMessage="No hardware found"
                    maxDropdownHeight={420}
                    leftSection={<Cpu size={17} />}
                    comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                  />
                </label>
                <label className="field-group">
                  <span>Num GPUs</span>
                  <small>{mode === 'inference' ? 'Devices for parallel inference.' : 'Devices for parallel fine-tuning.'}</small>
                  <NumberInput min={1} max={16} value={numGpus} onChange={(value) => setNumGpus(clamp(Number(value), 1, 16))} clampBehavior="strict" />
                </label>
              </div>

              <Select
                className="field-group runtime-field"
                label="Runtime Profile"
                description="Calibrates usable memory, runtime overhead, and token/sec estimates."
                data={runtimeOptions}
                value={runtimeProfile}
                onChange={(value) => value && setRuntimeProfile(value)}
                leftSection={<Activity size={17} />}
                comboboxProps={{ withinPortal: true, shadow: 'lg' }}
              />

              <div className="quant-matrix-toggle">
                <Button
                  variant="subtle"
                  color="violet"
                  leftSection={<Gauge size={17} />}
                  onClick={() => setShowQuantizationMatrix((value) => !value)}
                >
                  {showQuantizationMatrix ? 'Hide quantization options' : 'Show quantization options'}
                </Button>
              </div>

              {showQuantizationMatrix && (
                <QuantizationMatrix
                  model={selectedModel}
                  hardware={hardware}
                  rows={quantizationRows}
                  selectedQuant={weightQuant}
                  onSelect={setWeightQuant}
                />
              )}

              {showAdvanced && (
                <div className="advanced-grid">
                  <label className="field-group">
                    <span>Custom VRAM / GPU</span>
                    <small>Leave blank to use selected hardware.</small>
                    <NumberInput min={1} step={1} value={customVram} onChange={(value) => setCustomVram(value === '' ? '' : String(value))} placeholder={`${selectedHardware.memoryGb} GB`} />
                  </label>
                  <label className="field-group">
                    <span>Enable Offloading to CPU/RAM or NVMe</span>
                    <small>Reduces resident KV pressure and adds latency.</small>
                    <Switch checked={offload} onChange={(event) => setOffload(event.currentTarget.checked)} label={offload ? 'Enabled' : 'Disabled'} color="violet" size="md" />
                  </label>
                </div>
              )}

              <div className="divider" />

              <div className="input-heading">
                <strong>Input Parameters</strong>
                <span>Slider</span>
              </div>
              <ParameterSlider label="Batch Size" help={mode === 'inference' ? 'Inputs processed simultaneously per step. Affects throughput and latency.' : 'Training micro-batch size. Larger batches increase activation and optimizer pressure.'} min={1} max={32} marks={[1, 8, 16, 32]} value={batchSize} onChange={setBatchSize} />
              <ParameterSlider label="Sequence Length" help={mode === 'inference' ? 'Max tokens per input; impacts KV cache and activations.' : 'Training context length. Long sequences increase activation memory sharply.'} min={512} max={selectedModel.contextLength || 32768} step={512} marks={dynamicSequenceMarks(selectedModel.contextLength || 32768)} value={effectiveSequenceLength} onChange={setSequenceLength} formatter={formatNumber} />
              {mode === 'inference' && <ParameterSlider label="Concurrent Users" help="Number of users running inference simultaneously." min={1} max={32} marks={[1, 4, 8, 16, 32]} value={concurrentUsers} onChange={setConcurrentUsers} />}

              <ModelFacts model={selectedModel} />
              <EvidencePanel model={selectedModel} hardware={selectedHardware} benchmarks={matchingBenchmarks} />
            </section>

            <aside className="result-stack" aria-label="Results">
              <section className="result-card hero-result">
                <div className="result-heading">
                  <div>
                    <h2>{mode === 'inference' ? 'Performance & Memory Results' : 'Fine-tuning Memory Results'}</h2>
                    <p>{fit.label} for {hardware.name}</p>
                  </div>
                  <span className={`status-chip ${fit.tone}`}>{fit.label}</span>
                </div>

                <div className="donut-row">
                  <Donut percent={result.utilization} tone={fit.tone} />
                  <div className="headline-metrics">
                    <strong>{formatGb(result.totalRequiredGb)}</strong>
                    <span>of {formatGb(result.availableGb)} usable VRAM</span>
                    <em>{result.headroomGb >= 0 ? `${formatGb(result.headroomGb)} headroom` : `${formatGb(Math.abs(result.headroomGb))} over budget`}</em>
                  </div>
                </div>

                <div className="metric-list">
                  {mode === 'inference' ? (
                    <>
                      <Metric icon={<Gauge size={18} />} label="Decode Speed" value={canRun ? `~${formatNumber(getDecodeTokSec(result))} tok/sec` : 'Not runnable'} hint={canRun ? `${formatNumber(result.speedLow)}-${formatNumber(result.speedHigh)} tok/sec expected band` : overBudgetHint(result)} />
                      <Metric icon={<Activity size={18} />} label="Prefill Speed" value={canRun ? `~${formatNumber(getPrefillTokSec(result))} tok/sec` : 'Unavailable'} hint={canRun ? 'Prompt processing before token generation starts' : 'No valid prefill estimate while memory is over budget.'} />
                      <Metric icon={<Zap size={18} />} label="Time to First Token" value={canRun ? `~${formatNumber(getTtftMs(result))} ms` : 'Unavailable'} hint={canRun ? `${result.queuePenaltyMs ? `${formatNumber(result.queuePenaltyMs)} ms queue penalty. ` : ''}${formatNumber(result.maxResidentSequences)} resident sequence${result.maxResidentSequences === 1 ? '' : 's'} estimated` : 'No valid latency estimate while memory is over budget.'} />
                      <Metric icon={<Zap size={18} />} label="Total Throughput" value={canRun ? `~${formatNumber(result.totalThroughput)} tok/sec` : 'Not runnable'} hint={canRun ? `${numGpus} device${numGpus > 1 ? 's' : ''}, ${concurrentUsers} active user${concurrentUsers > 1 ? 's' : ''}` : overBudgetHint(result)} />
                    </>
                  ) : (
                    <>
                      <Metric icon={<Gauge size={18} />} label="Training Step Rate" value={canRun ? `~${result.stepsPerHour.toFixed(1)} steps/hr` : 'Not runnable'} hint={canRun ? 'Approximate memory-bound adapter/full-SFT pace' : overBudgetHint(result)} />
                      <Metric icon={<Activity size={18} />} label="Tokens per Step" value={formatNumber(result.tokensPerStep)} hint="Batch size multiplied by context length" />
                      <Metric icon={<Zap size={18} />} label="Effective Tokens/hr" value={canRun ? `~${formatNumber(result.trainingTokensHour)}` : 'Not runnable'} hint={canRun ? `${numGpus} device${numGpus > 1 ? 's' : ''}, ${trainingMethod.toUpperCase()} method` : overBudgetHint(result)} />
                    </>
                  )}
                </div>

                <div className="profile-summary">
                  <h3>{selectedModel.provider} {selectedModel.name}</h3>
                  <div>
                    <span>{weightQuant.toUpperCase()} weights</span>
                    <span>{mode === 'inference' ? `${kvQuant.toUpperCase()} KV` : trainingMethod.toUpperCase()}</span>
                    <span>{selectedModel.attentionStructure?.toUpperCase() || 'ATTN'}</span>
                    <span>{selectedModel.positionEmbedding?.toUpperCase() || 'POS'}</span>
                  </div>
                  <p>Mode: {mode === 'inference' ? 'Inference' : 'Fine-tuning'} | Batch: {batchSize}</p>
                </div>

                {mode === 'inference' && <RecommendationPanel recommendations={fitRecommendations} onApply={applyRecommendation} />}

                <EstimatorDetails result={result} mode={mode} model={selectedModel} />
                {mode === 'inference' && <PerformanceBasis model={selectedModel} />}
              </section>

              {mode === 'inference' && <ObservedBenchmarks benchmarks={matchingBenchmarks} activeRuntime={result.runtimeProfile} result={result} sequenceLength={effectiveSequenceLength} />}

              <section className="result-card">
                <h2>Memory Allocation</h2>
                <StackedBar segments={result.allocation} total={result.totalRequiredGb} />
                <div className="allocation-grid">
                  {result.allocation.map((item) => (
                    <div className="allocation-item" key={item.key}>
                      <span>{item.label}</span>
                      <strong>{formatGb(item.gb)}</strong>
                      <em style={{ color: item.color }}>{Math.round((item.gb / Math.max(result.totalRequiredGb, 0.01)) * 100)}%</em>
                      <i style={{ background: item.color }} />
                    </div>
                  ))}
                </div>
              </section>

              <section className="result-card economy-card">
                <Metric icon={<Cpu size={18} />} label="Power Draw" value={`${Math.round(result.powerW)}W`} />
                <Metric icon={<Database size={18} />} label="Cost per Hour" value={`$${result.costPerHour.toFixed(3)}`} />
                <Metric icon={<Sparkles size={18} />} label="CO₂/day" value={`${result.co2KgDay.toFixed(2)} kg`} />
              </section>
            </aside>
          </div>

          <CompareProfiles
            mode={mode}
            profiles={compareProfiles}
            modelOptions={modelOptions}
            hardwareOptions={hardwareOptions}
            batchSize={batchSize}
            sequenceLength={sequenceLength}
            concurrentUsers={concurrentUsers}
            offload={offload}
            runtimeProfile={runtimeProfile}
            onAdd={addCompareProfile}
            onClone={cloneCompareProfile}
            onRemove={removeCompareProfile}
            onUpdate={updateCompareProfile}
          />

          <ModelEvidenceLibrary />
        </section>
      </div>
    </main>
  );
}

function EstimatorDetails({ result, mode, model }) {
  return (
    <div className="estimator-details" aria-label="Estimator calibration details">
      <div>
        <span>Runtime</span>
        <strong>{result.runtimeLabel}</strong>
        <em>{result.estimateConfidence}</em>
      </div>
      <div>
        <span>Compute Path</span>
        <strong>{formatNumber(result.computeParamsB)}B active</strong>
        <em>{result.computeParamSource}</em>
      </div>
      <div>
        <span>Effective Storage</span>
        <strong>{result.weightBpw.toFixed(2)} bpw</strong>
        <em>{model.paramsB ? `${formatNumber(model.paramsB)}B stored` : 'estimated stored params'}</em>
      </div>
      {mode === 'inference' && (
        <div>
          <span>KV Cache</span>
          <strong>{formatGb(result.kvGbPer1kTokens)}</strong>
          <em>per 1K tokens / user</em>
        </div>
      )}
    </div>
  );
}

function PerformanceBasis({ model }) {
  const profile = model.performance;
  if (!profile) return null;

  const observed = profile.observedSummary;
  const coverage = profile.observedMetricCoverage || {};
  const coverageItems = [
    coverage.decode ? 'decode' : null,
    coverage.prefill ? 'prefill' : null,
    coverage.ttft ? 'TTFT' : null
  ].filter(Boolean);

  return (
    <section className="performance-basis" aria-label="Performance estimate basis">
      <div className="performance-basis-heading">
        <h3>Performance Basis</h3>
        <span>{profile.basis}</span>
      </div>
      <div className="performance-basis-grid">
        <div>
          <span>Decode</span>
          <strong>{profile.decodeSpeedClass}</strong>
          <em>{formatNumber(profile.computeParamsB)}B active params · factor {profile.decodeWorkFactor}</em>
        </div>
        <div>
          <span>Prefill</span>
          <strong>{profile.prefillSpeedClass}</strong>
          <em>factor {profile.prefillWorkFactor} · {profile.longContextPenaltyClass}</em>
        </div>
        <div>
          <span>TTFT Risk</span>
          <strong>{profile.ttftRiskClass}</strong>
          <em>{formatGb(profile.kvCacheGbPer1kFp16)} KV / 1K tokens at FP16</em>
        </div>
        <div>
          <span>Observed Coverage</span>
          <strong>{profile.observedBenchmarkCount || 0} run{profile.observedBenchmarkCount === 1 ? '' : 's'}</strong>
          <em>{coverageItems.length ? coverageItems.join(', ') : 'no direct observed run yet'}</em>
        </div>
      </div>
      {observed?.decodeTokSec && (
        <p>
          Observed decode range: {formatNumber(observed.decodeTokSec.min)}-{formatNumber(observed.decodeTokSec.max)} tok/sec
          {observed.prefillTokSec ? ` · prefill ${formatNumber(observed.prefillTokSec.min)}-${formatNumber(observed.prefillTokSec.max)} tok/sec` : ''}
          {observed.ttftMs ? ` · TTFT ${formatNumber(observed.ttftMs.min)}-${formatNumber(observed.ttftMs.max)} ms` : ''}
        </p>
      )}
    </section>
  );
}

function RecommendationPanel({ recommendations, onApply }) {
  if (!recommendations.length) return null;

  return (
    <section className="recommendation-panel" aria-label="Fit recommendations">
      <div className="recommendation-heading">
        <h3>Fit Recommendations</h3>
        <span>{recommendations.length} option{recommendations.length === 1 ? '' : 's'}</span>
      </div>
      <div className="recommendation-list">
        {recommendations.map((recommendation) => (
          <article key={recommendation.id} className="recommendation-item">
            <div>
              <strong>{recommendation.title}</strong>
              <p>{recommendation.body}</p>
            </div>
            {recommendation.actionLabel && (
              <Button size="xs" variant="light" color="violet" onClick={() => onApply(recommendation.action)}>
                {recommendation.actionLabel}
              </Button>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function ObservedBenchmarks({ benchmarks, activeRuntime, result, sequenceLength }) {
  if (!benchmarks.length) return null;

  const rows = nearestContextBenchmarks(benchmarks, sequenceLength, 10);
  const estimatedDecode = getDecodeTokSec(result);
  const estimatedPrefill = getPrefillTokSec(result);
  const estimatedTtft = getTtftMs(result);

  return (
    <section className="result-card observed-card" aria-label="Observed benchmarks">
      <div className="observed-heading">
        <div>
          <h2>Observed Benchmarks</h2>
          <p>Source-backed runs nearest to the selected context, compared with the current estimate.</p>
        </div>
        <span>{rows.length} of {benchmarks.length} run{benchmarks.length === 1 ? '' : 's'}</span>
      </div>

      <div className="observed-table">
        <div className="observed-row observed-header">
          <span>Engine</span>
          <span>Run</span>
          <span>Observed</span>
          <span>Estimate</span>
          <span>Prefill / TTFT</span>
        </div>
        {rows.map((benchmark) => {
          const active = benchmark.engine === activeRuntime;
          const deltaLabel = formatDeltaPercent(benchmark.generationTokSec, estimatedDecode);
          const runLabel = [
            benchmark.quantization,
            benchmark.contextTokens ? formatContextLabel(benchmark.contextTokens) : null,
            benchmark.batchSize ? `batch ${benchmark.batchSize}` : null
          ].filter(Boolean).join(' · ');
          return (
            <a className={`observed-row${active ? ' active' : ''}`} href={benchmark.sourceUrl} target="_blank" rel="noreferrer" key={benchmark.id}>
              <span>
                <strong>{benchmark.engineLabel}</strong>
                <em>{benchmark.confidence}</em>
              </span>
              <span>{runLabel || benchmark.hardwareName}</span>
              <span>
                <strong>{formatNumber(benchmark.generationTokSec)} tok/s</strong>
                {benchmark.tokensPerDollar && <em>{formatNumber(benchmark.tokensPerDollar)} tok/$</em>}
              </span>
              <span>
                <strong>{formatNumber(estimatedDecode)} tok/s</strong>
                {deltaLabel && <em className={benchmark.generationTokSec >= estimatedDecode ? 'positive' : 'negative'}>{deltaLabel}</em>}
              </span>
              <span>
                {benchmark.prefillTokSec ? `${formatNumber(benchmark.prefillTokSec)} pp tok/s` : `${formatNumber(estimatedPrefill)} pp tok/s est.`}
                <em>{benchmark.ttftMs ? `${formatNumber(benchmark.ttftMs)} ms TTFT` : `${formatNumber(estimatedTtft)} ms TTFT est.`}</em>
                {benchmark.peakMemoryGb && <em>{formatGb(benchmark.peakMemoryGb)} peak</em>}
              </span>
            </a>
          );
        })}
      </div>
    </section>
  );
}

function QuantizationMatrix({ model, hardware, rows, selectedQuant, onSelect }) {
  const fitRows = rows.filter((row) => row.result.headroomGb >= 0);
  const bestRow = (fitRows.length ? fitRows : rows)
    .slice()
    .sort((a, b) => {
      if (a.option.score !== b.option.score) return b.option.score - a.option.score;
      return a.result.totalRequiredGb - b.result.totalRequiredGb;
    })[0];
  const maxVram = Math.max(...rows.map((row) => row.result.totalRequiredGb), 1);
  const runtimeName = rows[0]?.result.runtimeLabel || 'current runtime';

  return (
    <section className="quant-matrix" aria-label="Quantization options">
      <div className="quant-matrix-heading">
        <div>
          <h3>Quantization options</h3>
          <p>
            How {model.name} fits across weight formats on {hardware.vendor} {hardware.name} ({formatGb(rows[0]?.result.availableGb || 0)} usable).
          </p>
        </div>
        <span>{runtimeName}</span>
      </div>

      <div className="quant-table-wrap">
        <table className="quant-table">
          <thead>
            <tr>
              <th>Quant</th>
              <th>Bits</th>
              <th>Memory Usage</th>
              <th>Quality</th>
              <th>Fit</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ option, result, fit, grade }) => {
              const selected = option.id === selectedQuant;
              const best = option.id === bestRow?.option.id;
              const barWidth = `${Math.max(5, Math.min(100, (result.totalRequiredGb / maxVram) * 100))}%`;

              return (
                <tr key={option.id} className={`${fit.tone}${selected ? ' selected' : ''}`} onClick={() => onSelect(option.id)}>
                  <td>
                    <button
                      type="button"
                      className="quant-name"
                      onClick={(event) => {
                        event.stopPropagation();
                        onSelect(option.id);
                      }}
                    >
                      <strong>{option.label}</strong>
                      {best && <em>Best fit</em>}
                      {selected && <em>Selected</em>}
                    </button>
                    <small>{option.description}</small>
                  </td>
                  <td>{option.bits}</td>
                  <td>
                    <strong>{formatGb(result.totalRequiredGb)}</strong>
                    <small>{Math.round(result.utilization)}% of usable memory</small>
                    <span className="quant-bar" aria-hidden="true">
                      <i style={{ width: barWidth }} />
                    </span>
                  </td>
                  <td>
                    <strong>{option.quality}</strong>
                    <em className={`mini-badge ${getQualityTone(option.score)}`}>{option.score}</em>
                  </td>
                  <td>
                    <span className={`mini-badge ${grade.tone}`}>{grade.grade} {grade.score}</span>
                    <small>{fit.label}</small>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CompareProfiles({ mode, profiles, modelOptions, hardwareOptions, batchSize, sequenceLength, concurrentUsers, offload, runtimeProfile, onAdd, onClone, onRemove, onUpdate }) {
  const [viewMode, setViewMode] = useState('cards');
  const cards = profiles.map((profile) => {
    const model = models.find((item) => item.id === Number(profile.modelId)) || defaultModel;
    const hardware = hardwareProfiles.find((item) => item.id === profile.hardwareId) || defaultHardware;
    const modelContextLimit = model.contextLength || 32768;
    const configuredSequenceLength = clamp(Number(profile.sequenceLength ?? sequenceLength), 512, modelContextLimit);
    const effectiveSequenceLength = Math.min(configuredSequenceLength, modelContextLimit);
    const contextOptions = contextOptionsForModel(modelContextLimit, effectiveSequenceLength);
    const result = calculateInference({
      mode,
      model,
      hardware,
      weightQuant: profile.weightQuant,
      kvQuant: profile.kvQuant,
      trainingMethod: profile.trainingMethod,
      numGpus: profile.numGpus,
      batchSize,
      sequenceLength: effectiveSequenceLength,
      concurrentUsers,
      offload,
      runtimeProfile
    });
    const quantizationRows = getQuantizationRows({
      mode,
      model,
      hardware,
      kvQuant: profile.kvQuant,
      trainingMethod: profile.trainingMethod,
      numGpus: profile.numGpus,
      batchSize,
      sequenceLength: effectiveSequenceLength,
      concurrentUsers,
      offload,
      runtimeProfile
    });
    const weightSelectOptions = getWeightSelectOptions(quantizationRows);
    return { profile, model, hardware, result, fit: getFitStatus(result.utilization), effectiveSequenceLength, modelContextLimit, contextOptions, weightSelectOptions };
  });
  const bestMemory = Math.min(...cards.map((card) => card.result.totalRequiredGb));
  const runnableCards = cards.filter((card) => isRunnableResult(card.result));
  const bestSpeed = runnableCards.length
    ? Math.max(...runnableCards.map((card) => (mode === 'inference' ? getDecodeTokSec(card.result) : card.result.trainingTokensHour)))
    : null;
  const kvSelectOptions = kvOptions.map((option) => ({ value: option.id, label: option.label }));
  const trainingSelectOptions = trainingOptions.map((option) => ({ value: option.id, label: option.label }));

  return (
    <section className="compare-panel" aria-label="Comparison profiles">
      <div className="compare-heading">
        <div>
          <h2>Compare Profiles</h2>
          <p>Select models, hardware, quantization, GPU count, and context per profile. Batch, offloading, and concurrent-user inputs stay shared.</p>
        </div>
        <div className="compare-actions">
          <Button className="add-profile-button" variant="light" color="violet" leftSection={<Plus size={17} />} onClick={onAdd} disabled={profiles.length >= 6}>
            Add Profile
          </Button>
          <SegmentedControl
            value={viewMode}
            onChange={setViewMode}
            data={[
              { value: 'cards', label: 'Cards' },
              { value: 'table', label: 'Table' }
            ]}
            size="sm"
            radius="md"
          />
        </div>
      </div>

      {viewMode === 'table' ? (
        <div className="compare-table-wrap">
          <table className="compare-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Hardware</th>
                <th>Weights</th>
                <th>{mode === 'inference' ? 'KV Cache' : 'Training'}</th>
                <th>GPUs</th>
                <th>Context</th>
                <th>VRAM</th>
                <th>{mode === 'inference' ? 'Speed' : 'Tokens/hr'}</th>
                <th>Headroom</th>
                <th>Status</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {cards.map(({ profile, model, result, fit, effectiveSequenceLength, modelContextLimit, contextOptions, weightSelectOptions }) => {
                const speedValue = mode === 'inference' ? getDecodeTokSec(result) : result.trainingTokensHour;
                const isBestMemory = result.totalRequiredGb === bestMemory;
                const canRunProfile = isRunnableResult(result);
                const isBestSpeed = canRunProfile && speedValue === bestSpeed;

                return (
                  <tr key={profile.id} className={fit.tone}>
                    <td>
                      <Select
                        aria-label={`Model for ${model.name}`}
                        searchable
                        data={modelOptions}
                        value={profile.modelId}
                        onChange={(value) => {
                          if (value) {
                            const nextModel = models.find((item) => item.id === Number(value)) || defaultModel;
                            onUpdate(profile.id, {
                              modelId: value,
                              sequenceLength: Math.min(effectiveSequenceLength, nextModel.contextLength || 32768)
                            });
                          }
                        }}
                        maxDropdownHeight={320}
                        comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                      />
                    </td>
                    <td>
                      <Select
                        aria-label={`Hardware for ${model.name}`}
                        searchable
                        data={hardwareOptions}
                        value={profile.hardwareId}
                        onChange={(value) => value && onUpdate(profile.id, { hardwareId: value })}
                        maxDropdownHeight={360}
                        comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                      />
                    </td>
                    <td>
                      <Select
                        aria-label={`Weights for ${model.name}`}
                        data={weightSelectOptions}
                        value={profile.weightQuant}
                        onChange={(value) => value && onUpdate(profile.id, { weightQuant: value })}
                        renderOption={renderWeightOption}
                        comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                      />
                    </td>
                    <td>
                      {mode === 'inference' ? (
                        <Select
                          aria-label={`KV cache for ${model.name}`}
                          data={kvSelectOptions}
                          value={profile.kvQuant}
                          onChange={(value) => value && onUpdate(profile.id, { kvQuant: value })}
                          comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                        />
                      ) : (
                        <Select
                          aria-label={`Training method for ${model.name}`}
                          data={trainingSelectOptions}
                          value={profile.trainingMethod}
                          onChange={(value) => value && onUpdate(profile.id, { trainingMethod: value })}
                          comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                        />
                      )}
                    </td>
                    <td className="table-number-cell">
                      <NumberInput
                        aria-label={`GPU count for ${model.name}`}
                        min={1}
                        max={16}
                        value={profile.numGpus}
                        onChange={(value) => onUpdate(profile.id, { numGpus: clamp(Number(value), 1, 16) })}
                        clampBehavior="strict"
                      />
                    </td>
                    <td className="table-context-cell">
                      <Select
                        aria-label={`Context length for ${model.name}`}
                        data={contextOptions}
                        value={String(effectiveSequenceLength)}
                        onChange={(value) => value && onUpdate(profile.id, { sequenceLength: clamp(Number(value), 512, modelContextLimit) })}
                        comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                      />
                    </td>
                    <td>
                      <strong>{formatGb(result.totalRequiredGb)}</strong>
                      {isBestMemory && <em className="table-note">Lowest</em>}
                    </td>
                    <td>
                      <strong>{formatRunRate(mode, result)}</strong>
                      {isBestSpeed && <em className="table-note">Fastest</em>}
                    </td>
                    <td className={result.headroomGb >= 0 ? 'positive' : 'negative'}>
                      {result.headroomGb >= 0 ? formatGb(result.headroomGb) : `-${formatGb(Math.abs(result.headroomGb))}`}
                    </td>
                    <td>
                      <span className={`mini-badge ${fit.tone}`}>{fit.label}</span>
                    </td>
                    <td>
                      <div className="profile-row-actions">
                        <Tooltip label="Clone profile" withArrow>
                          <ActionIcon variant="subtle" color="gray" aria-label={`Clone ${model.name} comparison profile`} onClick={() => onClone(profile.id)} disabled={profiles.length >= 6}>
                            <Copy size={17} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Remove profile" withArrow>
                          <ActionIcon variant="subtle" color="gray" aria-label={`Remove ${model.name} comparison profile`} onClick={() => onRemove(profile.id)} disabled={profiles.length <= 1}>
                            <Trash2 size={17} />
                          </ActionIcon>
                        </Tooltip>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="profile-card-grid">
        {cards.map(({ profile, model, hardware, result, fit, effectiveSequenceLength, modelContextLimit, contextOptions, weightSelectOptions }) => {
          const speedValue = mode === 'inference' ? getDecodeTokSec(result) : result.trainingTokensHour;
          const isBestMemory = result.totalRequiredGb === bestMemory;
          const canRunProfile = isRunnableResult(result);
          const isBestSpeed = canRunProfile && speedValue === bestSpeed;

          return (
            <article className={`compare-card ${fit.tone}`} key={profile.id}>
              <header>
                <div>
                  <strong>{model.name}</strong>
                  <span>{model.provider} · {hardware.name}</span>
                </div>
                <div className="profile-row-actions">
                  <Tooltip label="Clone profile" withArrow>
                    <ActionIcon variant="subtle" color="gray" aria-label={`Clone ${model.name} comparison profile`} onClick={() => onClone(profile.id)} disabled={profiles.length >= 6}>
                      <Copy size={17} />
                    </ActionIcon>
                  </Tooltip>
                  <Tooltip label="Remove profile" withArrow>
                    <ActionIcon variant="subtle" color="gray" aria-label={`Remove ${model.name} comparison profile`} onClick={() => onRemove(profile.id)} disabled={profiles.length <= 1}>
                      <Trash2 size={17} />
                    </ActionIcon>
                  </Tooltip>
                </div>
              </header>

              <div className="compare-controls">
                <Select
                  label="Model"
                  searchable
                  data={modelOptions}
                  value={profile.modelId}
                  onChange={(value) => {
                    if (value) {
                      const nextModel = models.find((item) => item.id === Number(value)) || defaultModel;
                      onUpdate(profile.id, {
                        modelId: value,
                        sequenceLength: Math.min(effectiveSequenceLength, nextModel.contextLength || 32768)
                      });
                    }
                  }}
                  maxDropdownHeight={320}
                  comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                />
                <Select
                  label="Hardware"
                  searchable
                  data={hardwareOptions}
                  value={profile.hardwareId}
                  onChange={(value) => value && onUpdate(profile.id, { hardwareId: value })}
                  maxDropdownHeight={360}
                  comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                />
                <Select
                  label="Weights"
                  data={weightSelectOptions}
                  value={profile.weightQuant}
                  onChange={(value) => value && onUpdate(profile.id, { weightQuant: value })}
                  renderOption={renderWeightOption}
                  comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                />
                {mode === 'inference' ? (
                  <Select
                    label="KV Cache"
                    data={kvOptions.map((option) => ({ value: option.id, label: option.label }))}
                    value={profile.kvQuant}
                    onChange={(value) => value && onUpdate(profile.id, { kvQuant: value })}
                    comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                  />
                ) : (
                  <Select
                    label="Training"
                    data={trainingOptions.map((option) => ({ value: option.id, label: option.label }))}
                    value={profile.trainingMethod}
                    onChange={(value) => value && onUpdate(profile.id, { trainingMethod: value })}
                    comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                  />
                )}
                <NumberInput
                  label="GPUs"
                  min={1}
                  max={16}
                  value={profile.numGpus}
                  onChange={(value) => onUpdate(profile.id, { numGpus: clamp(Number(value), 1, 16) })}
                  clampBehavior="strict"
                />
                <Select
                  label="Context"
                  data={contextOptions}
                  value={String(effectiveSequenceLength)}
                  onChange={(value) => value && onUpdate(profile.id, { sequenceLength: clamp(Number(value), 512, modelContextLimit) })}
                  comboboxProps={{ withinPortal: true, shadow: 'lg' }}
                />
              </div>

              <div className="compare-metrics">
                <MetricLine label="VRAM" value={formatGb(result.totalRequiredGb)} badge={isBestMemory ? 'Lowest' : fit.label} tone={fit.tone} />
                <MetricLine label={mode === 'inference' ? 'Speed' : 'Tokens/hr'} value={formatRunRate(mode, result)} badge={isBestSpeed ? 'Fastest' : null} tone={canRunProfile ? 'good' : 'bad'} />
                <MetricLine label="Headroom" value={result.headroomGb >= 0 ? formatGb(result.headroomGb) : `-${formatGb(Math.abs(result.headroomGb))}`} tone={result.headroomGb >= 0 ? 'good' : 'bad'} />
                <MetricLine label="Context Used" value={formatNumber(effectiveSequenceLength)} />
              </div>
            </article>
          );
        })}
        </div>
      )}
    </section>
  );
}

function MetricLine({ label, value, badge, tone = 'neutral' }) {
  return (
    <div className="metric-line">
      <span>{label}</span>
      <strong>{value}</strong>
      {badge && <em className={`mini-badge ${tone}`}>{badge}</em>}
    </div>
  );
}

function getModelEvidenceItems(model) {
  const fieldEvidence = Object.entries(model.fieldEvidence || {}).map(([field, item]) => ({
    ...item,
    kind: 'docs',
    label: field,
    url: item.url || item.quoteUrl
  }));
  const methodEvidence = (model.performance?.methodEvidenceIds || [])
    .map((id) => performanceSources.find((source) => source.id === id))
    .filter(Boolean);
  return uniqueEvidence([...(model.evidence || []), ...fieldEvidence, ...methodEvidence]);
}

function ModelEvidenceLibrary() {
  const [evidenceQuery, setEvidenceQuery] = useState('');
  const benchmarkBySlug = useMemo(() => {
    return observedBenchmarks.reduce((acc, benchmark) => {
      acc[benchmark.modelSlug] = acc[benchmark.modelSlug] || [];
      acc[benchmark.modelSlug].push(benchmark);
      return acc;
    }, {});
  }, []);
  const rows = useMemo(() => {
    const term = evidenceQuery.trim().toLowerCase();
    return models
      .map((model) => {
        const benchmarks = benchmarkBySlug[model.slug] || [];
        const evidence = getModelEvidenceItems(model);
        const coverage = model.performance?.observedMetricCoverage || {};
        const searchText = [
          model.provider,
          model.name,
          model.family,
          model.slug,
          model.architecture,
          model.performance?.decodeSpeedClass,
          model.performance?.ttftRiskClass,
          benchmarks.map((benchmark) => benchmark.engineLabel).join(' ')
        ].filter(Boolean).join(' ').toLowerCase();
        return { model, benchmarks, evidence, coverage, searchText };
      })
      .filter((row) => !term || row.searchText.includes(term))
      .sort((a, b) => {
        const benchmarkDelta = b.benchmarks.length - a.benchmarks.length;
        if (benchmarkDelta !== 0) return benchmarkDelta;
        return compareModelsForSelect(a.model, b.model);
      });
  }, [benchmarkBySlug, evidenceQuery]);

  return (
    <section className="evidence-library" aria-label="All model evidence and benchmark library">
      <div className="evidence-library-heading">
        <div>
          <h2>Evidence & Benchmark Library</h2>
          <p>All model citations, config-field proofs, performance-method sources, and observed benchmark coverage in one searchable table.</p>
        </div>
        <div className="evidence-library-search">
          <Search size={17} />
          <input value={evidenceQuery} onChange={(event) => setEvidenceQuery(event.target.value)} placeholder="Search model, provider, family, engine..." />
        </div>
      </div>

      <div className="evidence-library-summary">
        <MetricLine label="Models" value={formatNumber(models.length)} />
        <MetricLine label="Visible" value={formatNumber(rows.length)} />
        <MetricLine label="Benchmarked" value={formatNumber(models.filter((model) => (benchmarkBySlug[model.slug] || []).length).length)} />
        <MetricLine label="Benchmark Runs" value={formatNumber(observedBenchmarks.length)} />
      </div>

      <div className="evidence-library-table-wrap">
        <table className="evidence-library-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Performance Basis</th>
              <th>Observed Benchmarks</th>
              <th>Sources</th>
              <th>Citation Samples</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ model, benchmarks, evidence, coverage }) => {
              const coverageItems = [
                coverage.decode ? 'decode' : null,
                coverage.prefill ? 'prefill' : null,
                coverage.ttft ? 'TTFT' : null
              ].filter(Boolean);
              const observed = model.performance?.observedSummary;
              const sourceKinds = [...new Set(evidence.map((item) => evidenceKindLabels[item.kind] || item.kind))].slice(0, 5);

              return (
                <tr key={model.slug}>
                  <td>
                    <strong>{model.provider} {model.name}</strong>
                    <em>{model.family} · {model.paramsB ? `${formatNumber(model.paramsB)}B` : 'unknown params'} · {model.architecture}</em>
                  </td>
                  <td>
                    <span className="library-chip-row">
                      <span className="library-chip">decode {model.performance?.decodeSpeedClass || 'n/a'}</span>
                      <span className="library-chip">prefill {model.performance?.prefillSpeedClass || 'n/a'}</span>
                      <span className="library-chip">TTFT {model.performance?.ttftRiskClass || 'n/a'}</span>
                    </span>
                    <em>{model.performance ? `${formatNumber(model.performance.computeParamsB)}B active · ${formatGb(model.performance.kvCacheGbPer1kFp16)} KV/1K` : 'No performance profile'}</em>
                  </td>
                  <td>
                    <strong>{benchmarks.length} run{benchmarks.length === 1 ? '' : 's'}</strong>
                    <em>{coverageItems.length ? coverageItems.join(', ') : 'estimator only'}</em>
                    {observed?.decodeTokSec && <small>{formatNumber(observed.decodeTokSec.min)}-{formatNumber(observed.decodeTokSec.max)} tok/sec decode</small>}
                  </td>
                  <td>
                    <strong>{evidence.length} cited source{evidence.length === 1 ? '' : 's'}</strong>
                    <em>{sourceKinds.join(', ')}</em>
                  </td>
                  <td>
                    <div className="library-citation-links">
                      {evidence.slice(0, 4).map((item) => (
                        <a href={item.url} target="_blank" rel="noreferrer" key={evidenceKey(item)}>
                          <span>{evidenceKindLabels[item.kind] || item.kind || 'Source'}</span>
                          <strong>{item.label}</strong>
                          <em>{item.quote}</em>
                        </a>
                      ))}
                      {benchmarks.slice(0, 2).map((benchmark) => (
                        <a href={benchmark.sourceUrl} target="_blank" rel="noreferrer" key={benchmark.id}>
                          <span>Benchmark</span>
                          <strong>{benchmark.engineLabel}</strong>
                          <em>{formatNumber(benchmark.generationTokSec)} tok/sec decode{benchmark.prefillTokSec ? ` · ${formatNumber(benchmark.prefillTokSec)} prefill` : ''}</em>
                        </a>
                      ))}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SegmentedField({ label, value, onChange, options }) {
  const selected = options.find((option) => option.id === value);
  return (
    <div className="segmented-field">
      <span className="control-label">{label}</span>
      <SegmentedControl
        fullWidth
        value={value}
        onChange={onChange}
        data={options.map((option) => ({ value: option.id, label: option.label }))}
        color="violet"
        radius="md"
        size="sm"
      />
      <small>{selected?.description}</small>
    </div>
  );
}

function SelectField({ label, value, onChange, options, selectData, renderOption, leftSection }) {
  const selected = options.find((option) => option.id === value);
  return (
    <div className="select-field">
      <span className="control-label">{label}</span>
      <Select
        value={value}
        onChange={(nextValue) => nextValue && onChange(nextValue)}
        data={selectData || options.map((option) => ({ value: option.id, label: option.label }))}
        renderOption={renderOption}
        leftSection={leftSection}
        maxDropdownHeight={380}
        comboboxProps={{ withinPortal: true, shadow: 'lg' }}
      />
      <small>{selected?.description}</small>
    </div>
  );
}

function ParameterSlider({ label, help, min, max, step = 1, marks, value, onChange, formatter = (x) => x }) {
  return (
    <div className="slider-field">
      <div>
        <span>{label}</span>
        <strong>{formatter(value)}</strong>
      </div>
      <small>{help}</small>
      <Slider
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        step={step}
        color="violet"
        radius="xl"
        size="sm"
        label={(nextValue) => formatter(nextValue)}
        marks={marks.map((mark) => ({ value: mark, label: formatter(mark) }))}
      />
    </div>
  );
}

function ModelFacts({ model }) {
  const facts = [
    ['Parameters', model.paramsB ? `${model.paramsB}B` : 'Unknown'],
    ['Context', model.contextLength ? formatNumber(model.contextLength) : 'Unknown'],
    ['Layers', model.layers || 'Unknown'],
    ['Architecture', model.architecture || 'Unknown'],
    ['Release', model.releaseDate || 'Unknown'],
    ['Weights', model.openWeights ? 'Open' : 'Restricted']
  ];

  return (
    <section className="model-facts">
      <h3>Model Profile</h3>
      <div>
        {facts.map(([label, value]) => (
          <p key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </p>
        ))}
      </div>
    </section>
  );
}

const evidenceKindLabels = {
  official: 'Official',
  docs: 'Docs',
  datasheet: 'Datasheet',
  paper: 'Paper',
  weights: 'Weights',
  benchmark: 'Benchmark',
  github: 'GitHub',
  reddit: 'Reddit',
  community: 'Community',
  pricing: 'Pricing'
};

function evidenceKey(item) {
  return `${item.kind || 'source'}:${item.url}:${item.label || ''}:${item.claim || ''}`;
}

function uniqueEvidence(items) {
  const seen = new Set();
  return items
    .filter((item) => item?.url && item?.label)
    .filter((item) => {
      const key = evidenceKey(item);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function EvidencePanel({ model, hardware, benchmarks }) {
  const modelEvidence = uniqueEvidence(model.evidence || []).slice(0, 8);
  const fieldLabels = {
    hiddenSize: 'Hidden size',
    layers: 'Layers',
    attentionHeads: 'Attention heads',
    kvHeads: 'KV heads',
    headDim: 'Head dimension',
    contextLength: 'Context length',
    slidingWindow: 'Sliding window'
  };
  const modelFieldEvidence = Object.entries(model.fieldEvidence || {}).map(([field, item]) => ({
    ...item,
    kind: 'docs',
    label: fieldLabels[field] || field,
    url: item.url || item.quoteUrl
  }));
  const hardwareEvidence = uniqueEvidence(hardware.evidence || []).slice(0, 8);
  const benchmarkEvidence = uniqueEvidence(
    benchmarks.flatMap((benchmark) => benchmark.evidence || [])
  ).slice(0, 8);
  const performanceMethodEvidence = uniqueEvidence(
    (model.performance?.methodEvidenceIds || [])
      .map((id) => performanceSources.find((source) => source.id === id))
      .filter(Boolean)
  ).slice(0, 8);
  const sections = [
    ['Model', modelEvidence],
    ['Model Fields', modelFieldEvidence],
    ['Performance Method', performanceMethodEvidence],
    ['Hardware', hardwareEvidence],
    ['Benchmarks', benchmarkEvidence]
  ].filter(([, items]) => items.length);

  if (!sections.length) return null;

  return (
    <section className="evidence-panel" aria-label="Evidence and source links">
      <div className="evidence-heading">
        <h3>Evidence</h3>
        <span>{sections.reduce((total, [, items]) => total + items.length, 0)} links</span>
      </div>
      <div className="evidence-sections">
        {sections.map(([title, items]) => (
          <div className="evidence-section" key={title}>
            <strong>{title}</strong>
            <div>
              {items.map((item) => (
                <a href={item.url} target="_blank" rel="noreferrer" key={evidenceKey(item)}>
                  <span>{evidenceKindLabels[item.kind] || item.kind || 'Source'}</span>
                  <strong>{item.label}</strong>
                  <em>{item.claim}</em>
                  <small>{item.quote}</small>
                  <ExternalLink size={14} />
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Donut({ percent, tone }) {
  const clamped = Math.max(0, Math.min(percent, 140));
  return (
    <div className={`donut ${tone}`} style={{ '--percent': `${Math.min(clamped, 100) * 3.6}deg` }}>
      <div>
        <strong>{Math.round(percent)}%</strong>
        <span>VRAM</span>
      </div>
    </div>
  );
}

function Metric({ icon, label, value, hint }) {
  return (
    <div className="metric">
      {icon}
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        {hint && <small>{hint}</small>}
      </div>
      <Info size={15} />
    </div>
  );
}

function StackedBar({ segments, total }) {
  return (
    <div className="stacked-bar" aria-label="Memory allocation">
      {segments.map((segment) => (
        <span key={segment.key} style={{ width: `${Math.max(3, (segment.gb / Math.max(total, 0.01)) * 100)}%`, background: segment.color }}>
          {segment.gb / total > 0.12 ? `${Math.round((segment.gb / total) * 100)}%` : ''}
        </span>
      ))}
    </div>
  );
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function dynamicSequenceMarks(max) {
  const safeMax = Math.max(2048, max || 32768);
  const marks = [safeMax / 16, safeMax / 4, safeMax / 2, safeMax]
    .map((value) => Math.max(512, Math.round(value / 512) * 512));
  return [...new Set(marks)];
}

createRoot(document.getElementById('root')).render(
  <MantineProvider
    theme={{
      primaryColor: 'violet',
      defaultRadius: 'md',
      fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      headings: {
        fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
      }
    }}
  >
    <App />
  </MantineProvider>
);
