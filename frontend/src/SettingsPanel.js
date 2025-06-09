import React, { useState, useEffect } from 'react';
import './css/SettingsPanel.css';

const requiresModelDir = ["Llama.cpp"];
const backendDirName = { "Ollama": "ollama", "HuggingFace": "huggingface", "Llama.cpp": "llamacpp" };
const modelNamesByBackend = {
  "Ollama": ["dolphin-mistral", "dolphin-llama3:8b", "huihui_ai/dolphin3-abliterated:latest", "sam860/dolphin3-qwen2.5:3b", "hammerai/mistral-nemo-uncensored"],
  "HuggingFace": ["cognitivecomputations/Dolphin3.0-Qwen2.5-3b"],
  "Llama.cpp": ["Dolphin3.0-Qwen2.5-3b.i1-Q4_K_M.gguf", "ggml-model-Q8_0.gguf"]
};

export const defaultSettings = {
  SwarmUI_Port: "7801",
  backend: "Llama.cpp",
  model_dir: "C:/Users/admin/Documents/React_Agent_AI/backend/models/llamacpp",
  llm_model: "Dolphin3.0-Qwen2.5-3b.i1-Q4_K_M.gguf",
  use_vision: false,
  vision_backend: "Ollama",
  vision_model_dir: "C:/Users/admin/Documents/React_Agent_AI/backend/vision_models/llamacpp",
  vision_model: "",
  use_danbooru_transform: false,
  lora_settings: {
    temperature: 0.1,
    top_p: 0.9,
    top_k: 30,
    repetition_penalty: 0.9,
    max_new_tokens: 512,
  }
};

export default function SettingsPanel({ settings, setSettings }) {
  const [local, setLocal] = useState({
    ...defaultSettings,
    ...(settings || {})
  });
  // Ollama model state
  const [ollamaModels, setOllamaModels] = useState([]);
  const [ollamaVisionModels, setOllamaVisionModels] = useState([]);
  const [ollamaLoading, setOllamaLoading] = useState(false);
  const [ollamaVisionLoading, setOllamaVisionLoading] = useState(false);
  const [ollamaError, setOllamaError] = useState(null);
  // Llama.cpp model state
  const [llamaCppModels, setLlamaCppModels] = useState([]);
  const [llamaCppLoading, setLlamaCppLoading] = useState(false);
  const [llamaCppError, setLlamaCppError] = useState(null);

  // --- LLM Backend effect ---
  useEffect(() => {
    if (local.backend === "Ollama") {
      setOllamaLoading(true);
      setOllamaError(null);
      setOllamaModels([]);
      fetch('/api/tags')
        .then(res => res.json())
        .then(data => {
          const modelNames = data.models.map(m => m.name);
          setOllamaModels(modelNames);
          setOllamaLoading(false);
        })
        .catch(err => {
          setOllamaError('Fetch Error: ' + err);
          setOllamaLoading(false);
        });
    }
    if (local.backend === "Llama.cpp" && local.model_dir) {
      setLlamaCppLoading(true);
      setLlamaCppError(null);
      setLlamaCppModels([]);
      fetch(`http://localhost:8000/api/list-llamacpp-models?dir=${encodeURIComponent(local.model_dir)}`)
        .then(res => res.json())
        .then(data => {
          setLlamaCppModels(data.models || []);
          setLlamaCppLoading(false);
        })
        .catch(err => {
          setLlamaCppError('Failed to load models');
          setLlamaCppLoading(false);
        });
    }
  }, [local.backend, local.model_dir]);

  // --- Vision Backend effect ---
  useEffect(() => {
    if (local.vision_backend === "Ollama") {
      setOllamaVisionLoading(true);
      setOllamaVisionModels([]);
      setOllamaError(null);
      fetch('/api/tags')
        .then(res => res.json())
        .then(data => {
          const modelNames = data.models.map(m => m.name);
          // Fetch all /api/show in parallel
          return Promise.all(
            modelNames.map(name =>
              fetch('/api/show', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
              })
                .then(res => res.json())
                .then(data => ({ name, data }))
                .catch(() => null)
            )
          );
        })
        .then(results => {
          // Filter for vision-capable models
          const visionModels = results
            .filter(r => r && r.data && r.data.capabilities && r.data.capabilities.includes("vision"))
            .map(r => r.name);
          setOllamaVisionModels(visionModels);
          setOllamaVisionLoading(false);
        })
        .catch(err => {
          setOllamaError('Fetch Error: ' + err);
          setOllamaVisionLoading(false);
        });
    }
  }, [local.vision_backend]);

  function getDefaultModelDir(backend) {
    return `C:/Users/admin/Documents/React_Agent_AI/backend/models/${backendDirName[backend] || ""}`;
  }

  function getDefaultVisionModelDir(backend) {
    return `C:/Users/admin/Documents/React_Agent_AI/backend/vision_models/${backendDirName[backend] || ""}`;
  }

  const updateSetting = (field, value) => {
    const updated = { ...local, [field]: value };

    // Auto-update model dir
    if (field === "backend" && requiresModelDir.includes(value)) {
      updated.model_dir = getDefaultModelDir(value);
    }
    if (field === "vision_backend" && requiresModelDir.includes(value)) {
      updated.vision_model_dir = getDefaultVisionModelDir(value);
    }

    // Reset llm_model appropriately when backend changes
    if (field === "backend") {
      if (value === "HuggingFace") {
        updated.llm_model = "cognitivecomputations/Dolphin3.0-Qwen2.5-3b";
      } else if (value === "Ollama" && ollamaModels.length > 0) {
        updated.llm_model = ollamaModels[0];
      } else if (value === "Llama.cpp" && llamaCppModels.length > 0) {
        updated.llm_model = llamaCppModels[0];
      } else {
        updated.llm_model = "";
      }
    }

    setLocal(updated);
    setSettings(updated);
  };

  const updateLoraSetting = (field, value) => {
    const updated = {
      ...local,
      lora_settings: { ...local.lora_settings, [field]: value }
    };
    setLocal(updated);
    setSettings(updated);
  };

  const getModelOptions = (backend, modelDir) => {
    return modelNamesByBackend[backend] || [];
  };

  return (
    <div className="settings-panel-row">
      {/* Agent Settings */}
      <details className="settings-panel agent-settings" open>
        <summary style={{fontSize: "1.2rem", fontWeight: 600}}>Agent Settings</summary>
        <div className="setting-block">
          <label>
            SwarmUI Port:
            <input
              type="text"
              value={local.SwarmUI_Port}
              onChange={(e) => updateSetting("SwarmUI_Port", e.target.value)}
            />
          </label>
        </div>
        <div className="setting-block">
          <label>
            LLM Backend:
            <select
              value={local.backend}
              onChange={(e) => updateSetting("backend", e.target.value)}
            >
              <option>Ollama</option>
              <option>Llama.cpp</option>
              <option>HuggingFace</option>
            </select>
          </label>
        </div>
        {requiresModelDir.includes(local.backend) && (
          <div className="setting-block">
            <label>
              Model Directory:
              <input
                type="text"
                value={local.model_dir}
                onChange={(e) => updateSetting("model_dir", e.target.value)}
              />
            </label>
          </div>
        )}
        {local.backend !== "HuggingFace" && (
        <div className="setting-block">
          <label>
            LLM Model:
            <select
              value={local.llm_model}
              onChange={(e) => updateSetting("llm_model", e.target.value)}
            >
              {local.backend === "Llama.cpp" ? (
                llamaCppLoading ? (
                  <option>Loading...</option>
                ) : llamaCppError ? (
                  <option>{llamaCppError}</option>
                ) : (
                  llamaCppModels.map((m, i) => (
                    <option key={i} value={m}>{m}</option>
                  ))
                )
              ) : local.backend === "Ollama" ? (
                ollamaLoading ? (
                  <option>Loading...</option>
                ) : ollamaError ? (
                  <option>{ollamaError}</option>
                ) : (
                  ollamaModels.map((m,i)=>(<option key={i} value={m}>{m}</option>))
                )
              ) : (
                getModelOptions(local.backend, local.model_dir).map((m, i) => (
                  <option key={i} value={m}>{m}</option>
                ))
              )}
            </select>
          </label>
        </div>)}
        {local.backend === "HuggingFace" && (
          <div className="setting-block">
            <label>
              Model Repo:
              <input
                type="text"
                value={local.llm_model}
                onChange={(e) => updateSetting("llm_model", e.target.value)}
              />
            </label>
          </div>
        )}
        <div className="checkbox-row">
            <div className="setting-block">
              <label>
                Use Vision:
                <input
                  type="checkbox"
                  checked={local.use_vision}
                  onChange={(e) => updateSetting("use_vision", e.target.checked)}
                />
              </label>
            </div>
            {//<div className="setting-block">
             // <label>
             //   Use Danbooru LoRa:
             //   <input
             //     type="checkbox"
             //     checked={local.use_danbooru_transform}
             //     onChange={(e) => updateSetting("use_danbooru_transform", e.target.checked)}
             //   />
             // </label>
             //</div>
             }
        </div>
      </details>

        {/* Vision Settings */}
        {local.use_vision && (
          <details open className="settings-panel agent-settings">
            <summary style={{fontSize: "1.2rem", fontWeight: 600}}>Vision Settings</summary>
            <div className="setting-block">
              <label>
                Vision Backend:
                <select
                  value={local.vision_backend}
                  onChange={(e) => updateSetting("vision_backend", e.target.value)}
                >
                  <option>Ollama</option>
                </select>
              </label>
            </div>
            {requiresModelDir.includes(local.vision_backend) && (
              <div className="setting-block">
                <label>
                  Model Directory:
                  <input
                    type="text"
                    value={local.vision_model_dir}
                    onChange={(e) => updateSetting("vision_model_dir", e.target.value)}
                  />
                </label>
              </div>
            )}
            <div className="setting-block">
              <label>
                Vision Model:
                <select
                  value={local.vision_model}
                  onChange={(e) => updateSetting("vision_model", e.target.value)}
                >
                  {local.vision_backend !== "Ollama" ? (
                    getModelOptions(local.backend, local.model_dir).map((m, i) => (
                      <option key={i} value={m}>{m}</option>
                    ))
                  ) : ollamaVisionLoading ? (
                    <option>Loading...</option>
                  ) : ollamaError ? (
                    <option>{ollamaError}</option>
                  ) : (
                    ollamaVisionModels.map((m,i)=>(
                      <option key={i} value={m}>{m}</option>
                    ))
                  )}
                </select>
              </label>
            </div>
          </details>
        )}

        {/* LoRA Settings */}
        {local.use_danbooru_transform && (
          <details open className="settings-panel agent-settings">
            <summary style={{fontSize: "1.2rem", fontWeight: 600}}>LoRA Settings</summary>
            <div className="setting-block">
              <label>
                Temperature:
                <input
                  type="range"
                  min="0.1" max="2.0" step="0.1"
                  value={local.lora_settings.temperature}
                  onChange={(e) => updateLoraSetting("temperature", parseFloat(e.target.value))}
                />
                {local.lora_settings.temperature}
              </label>
            </div>
            <div className="setting-block">
              <label>
                Top P:
                <input
                  type="range"
                  min="0" max="1.0" step="0.01"
                  value={local.lora_settings.top_p}
                  onChange={(e) => updateLoraSetting("top_p", parseFloat(e.target.value))}
                />
                {local.lora_settings.top_p}
              </label>
            </div>
            <div className="setting-block">
              <label>
                Top K:
                <input
                  type="range"
                  min="0" max="100" step="1"
                  value={local.lora_settings.top_k}
                  onChange={(e) => updateLoraSetting("top_k", parseInt(e.target.value))}
                />
                {local.lora_settings.top_k}
              </label>
            </div>
            <div className="setting-block">
              <label>
                Repetition Penalty:
                <input
                  type="range"
                  min="0.0" max="2.0" step="0.025"
                  value={local.lora_settings.repetition_penalty}
                  onChange={(e) => updateLoraSetting("repetition_penalty", parseFloat(e.target.value))}
                />
                {local.lora_settings.repetition_penalty}
              </label>
            </div>
            <div className="setting-block">
              <label>
                Max New Tokens:
                <input
                  type="range"
                  min="16" max="1024" step="8"
                  value={local.lora_settings.max_new_tokens}
                  onChange={(e) => updateLoraSetting("max_new_tokens", parseInt(e.target.value))}
                />
                {local.lora_settings.max_new_tokens}
              </label>
            </div>
          </details>
        )}
      </div>
  );
}