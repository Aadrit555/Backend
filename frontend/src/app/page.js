"use client";

import { useState, useRef, useEffect } from "react";
import { UploadCloud, CheckCircle2, DownloadCloud, Play, Sparkles, Search, X, Image as ImageIcon, Sliders, ArrowRight } from "lucide-react";
import CustomVisionStudio from "../components/CustomVisionStudio";

export default function BeginnerPage() {
  const [projectId, setProjectId] = useState("proj_init");

  useEffect(() => {
    setProjectId(`proj_${Math.random().toString(36).substr(2, 9)}`);
  }, []);

  const [modelCandidate, setModelCandidate] = useState("autogluon_best");
  const [activeTab, setActiveTab] = useState("tabular");
  const [visionSubMode, setVisionSubMode] = useState("classifier"); // "classifier" | "detection"
  const [files, setFiles] = useState([]);
  const [manifest, setManifest] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isBuilding, setIsBuilding] = useState(false);

  // Status polling state
  const [experimentId, setExperimentId] = useState(null);
  const [experimentBackend, setExperimentBackend] = useState(null);
  const [statusLogs, setStatusLogs] = useState([]);
  const [finalMetrics, setFinalMetrics] = useState(null);

  // Chat State
  const [ragQuery, setRagQuery] = useState("");
  const [ragChat, setRagChat] = useState([]);
  const [isQuerying, setIsQuerying] = useState(false);

  // Hugging Face Hub Import state
  const [importedModels, setImportedModels] = useState([]);
  const [showHfModal, setShowHfModal] = useState(false);
  const [hfQuery, setHfQuery] = useState("");
  const [hfTask, setHfTask] = useState("");
  const [hfResults, setHfResults] = useState([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [hfImporting, setHfImporting] = useState(false);
  const [hfError, setHfError] = useState(null);

  // Vision Inference Studio state
  const [visionFile, setVisionFile] = useState(null);
  const [visionPreview, setVisionPreview] = useState(null);
  const [visionConf, setVisionConf] = useState(0.25);
  const [isDetecting, setIsDetecting] = useState(false);
  const [visionResult, setVisionResult] = useState(null);
  const [visionError, setVisionError] = useState(null);
  const visionInputRef = useRef(null);

  const fileInputRef = useRef(null);
  const logsEndRef = useRef(null);

  // Poll for status
  const fmtTime = (iso) => {
    if (!iso) return "--:--:--.---";
    try {
      const d = new Date(iso);
      return d.toTimeString().split(' ')[0] + '.' + String(d.getMilliseconds()).padStart(3, '0');
    } catch { return iso; }
  };

  // Load imported models on mount
  const fetchImportedModels = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/hf/imported");
      if (res.ok) {
        const data = await res.json();
        setImportedModels(data.imported_models || []);
      }
    } catch (e) {
      console.error("Failed to load imported models:", e);
    }
  };

  useEffect(() => {
    fetchImportedModels();
  }, []);

  useEffect(() => {
    // Reset state when switching tabs
    setFiles([]);
    setManifest(null);
    setIsUploading(false);
    setIsBuilding(false);
    setExperimentId(null);
    setExperimentBackend(null);
    setStatusLogs([]);
    setFinalMetrics(null);
    setRagQuery("");
    setRagChat([]);
    setVisionFile(null);
    setVisionPreview(null);
    setVisionResult(null);
    setVisionError(null);

    switch (activeTab) {
      case "tabular": setModelCandidate("autogluon_best"); break;
      case "llm": setModelCandidate("unsloth/Llama-3.2-1B-Instruct-bnb-4bit"); break;
      case "rag": setModelCandidate("rag_default"); break;
      case "vision": setModelCandidate("yolov8n"); break;
    }
  }, [activeTab]);

  useEffect(() => {
    if (!experimentId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/experiments/${experimentId}/status`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.log && data.log.length > 0) {
          setStatusLogs(prev => {
            const frontendEntries = prev.filter(e => e._frontend);
            const backendEntries = data.log.map(e => ({ ...e, _backend: true }));
            return [...frontendEntries, ...backendEntries];
          });
        }

        if (data.stage === "completed" || data.stage === "failed") {
          clearInterval(interval);
          setIsBuilding(false);
          if (data.stage === "completed") {
            fetchProjectData(projectId);
          }
        }
      } catch (err) {
        console.error("Failed to fetch status:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [experimentId, projectId]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [statusLogs]);

  const fetchProjectData = async (pid) => {
    try {
      const res = await fetch(`http://localhost:8000/api/projects/${pid}`);
      if (!res.ok) return;
      const data = await res.json();

      const exp = data.experiments?.find(e => e.id === experimentId || true);
      if (exp) {
        if (exp.metrics) setFinalMetrics(exp.metrics);
        if (exp.backend) setExperimentBackend(exp.backend);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add("border-[#00E5FF]");
    e.currentTarget.classList.remove("border-[#333333]");
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("border-[#00E5FF]");
    e.currentTarget.classList.add("border-[#333333]");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("border-[#00E5FF]");
    e.currentTarget.classList.add("border-[#333333]");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFiles = async (selectedFiles) => {
    setFiles(selectedFiles);
    setIsUploading(true);

    const formData = new FormData();
    formData.append("project_id", projectId);
    selectedFiles.forEach(f => formData.append("files", f));

    try {
      const res = await fetch("http://localhost:8000/api/ingest", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setManifest(data.manifest);
      } else {
        alert("Upload failed.");
      }
    } catch (err) {
      console.error(err);
      alert("Network error during upload.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleBuild = async () => {
    setIsBuilding(true);
    setStatusLogs([{ stage: "started", message: "Initiating expert build...", pct: 0, updated_at: new Date().toISOString(), _frontend: true }]);

    try {
      const res = await fetch("http://localhost:8000/api/expert_build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          pipeline_type: activeTab,
          expert_config: { model_candidates: [modelCandidate] }
        })
      });
      if (res.ok) {
        setStatusLogs(prev => [...prev, { stage: "planning", message: "Build initialized. Starting execution...", pct: 0, updated_at: new Date().toISOString(), _frontend: true }]);
        pollForExperiment();
      } else {
        setStatusLogs(prev => [...prev, { stage: "failed", message: "API error initiating expert build.", pct: 0, updated_at: new Date().toISOString(), _frontend: true }]);
        setIsBuilding(false);
      }
    } catch (err) {
      setStatusLogs(prev => [...prev, { stage: "failed", message: "Network error.", pct: 0, updated_at: new Date().toISOString(), _frontend: true }]);
      setIsBuilding(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!ragQuery.trim()) return;

    setRagChat(prev => [...prev, { role: "user", content: ragQuery }]);
    setIsQuerying(true);

    try {
      const isLLM = experimentBackend === "unsloth";
      const url = isLLM
        ? `http://localhost:8000/api/models/${experimentId}/chat`
        : `http://localhost:8000/api/experiments/${experimentId}/query_rag`;

      const payload = isLLM
        ? { prompt: ragQuery, max_tokens: 128 }
        : { query: ragQuery };

      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        const msgContent = isLLM ? data.response : data.answer;
        setRagChat(prev => [...prev, { role: "assistant", content: msgContent, citations: data.citations }]);
      } else {
        const errorData = await res.json().catch(() => ({}));
        const errorDetail = errorData.detail || `HTTP Error ${res.status}`;
        setRagChat(prev => [...prev, { role: "assistant", content: `Error: ${errorDetail}` }]);
      }
    } catch (err) {
      setRagChat(prev => [...prev, { role: "assistant", content: `Network Error: ${err.message}` }]);
    } finally {
      setIsQuerying(false);
      setRagQuery("");
    }
  };

  const pollForExperiment = () => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > 120) { // 2 mins timeout
        clearInterval(interval);
        setStatusLogs(prev => [...prev, { stage: "failed", message: "Timeout waiting for orchestrator.", pct: 0, updated_at: new Date().toISOString(), _frontend: true }]);
        setIsBuilding(false);
        return;
      }

      try {
        const res = await fetch(`http://localhost:8000/api/projects/${projectId}`);
        if (res.ok) {
          const data = await res.json();
          if (data.experiments && data.experiments.length > 0) {
            clearInterval(interval);
            const exp = data.experiments[0];
            if (exp.status === "failed") {
              let errStr = "Unknown orchestration error";
              try { errStr = JSON.parse(exp.config_json).error || errStr; } catch (e) { }
              setStatusLogs(prev => [...prev, { stage: "failed", message: `Orchestrator Failed: ${errStr}`, pct: 0, _frontend: true }]);
              setIsBuilding(false);
            } else {
              setExperimentId(exp.id);
              setExperimentBackend(exp.backend);
              setStatusLogs(prev => [...prev, { stage: "execution", message: `Experiment ${exp.id} created. Starting training...`, pct: 5, updated_at: new Date().toISOString(), _frontend: true }]);
            }
          }
        }
      } catch (e) {
        // ignore
      }
    }, 2000);
  };

  // Hugging Face Hub handlers
  const handleHfSearch = async (e) => {
    if (e) e.preventDefault();
    if (!hfQuery.trim()) return;
    setHfSearching(true);
    setHfError(null);
    try {
      const taskParam = hfTask ? `&task=${encodeURIComponent(hfTask)}` : "";
      const res = await fetch(`http://localhost:8000/api/hf/search?query=${encodeURIComponent(hfQuery)}${taskParam}&limit=15`);
      if (res.ok) {
        const data = await res.json();
        setHfResults(data.models || []);
      } else {
        setHfError("Failed to search Hugging Face Hub.");
      }
    } catch (err) {
      setHfError("Network error while searching Hugging Face.");
    } finally {
      setHfSearching(false);
    }
  };

  const handleHfImport = async (modelId) => {
    setHfImporting(true);
    setHfError(null);
    try {
      const res = await fetch("http://localhost:8000/api/hf/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: modelId,
          pipeline_type: activeTab,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        await fetchImportedModels();
        setModelCandidate(data.model?.id || modelId);
        setShowHfModal(false);
      } else {
        const err = await res.json().catch(() => ({}));
        setHfError(err.detail || "Failed to import model.");
      }
    } catch (err) {
      setHfError("Network error importing model: " + err.message);
    } finally {
      setHfImporting(false);
    }
  };

  // Vision handlers
  const handleVisionFileSelect = (file) => {
    if (!file) return;
    setVisionFile(file);
    setVisionResult(null);
    setVisionError(null);
    const reader = new FileReader();
    reader.onload = (e) => setVisionPreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleVisionSample = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Background
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, 640, 480);

    // Road
    ctx.fillStyle = "#334155";
    ctx.fillRect(0, 300, 640, 180);

    // Car body
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(150, 240, 260, 100);
    ctx.fillStyle = "#991b1b";
    ctx.fillRect(190, 180, 180, 70);

    // Car windows
    ctx.fillStyle = "#93c5fd";
    ctx.fillRect(200, 190, 75, 50);
    ctx.fillRect(285, 190, 75, 50);

    // Car wheels
    ctx.fillStyle = "#0f172a";
    ctx.beginPath();
    ctx.arc(210, 340, 32, 0, Math.PI * 2);
    ctx.arc(350, 340, 32, 0, Math.PI * 2);
    ctx.fill();

    // Pedestrian
    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(490, 230, 30, 90);
    ctx.fillStyle = "#fde047";
    ctx.beginPath();
    ctx.arc(505, 210, 16, 0, Math.PI * 2);
    ctx.fill();

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "sample_street_scene.jpg", { type: "image/jpeg" });
        handleVisionFileSelect(file);
      }
    }, "image/jpeg");
  };

  const handleRunVisionInference = async () => {
    if (!visionFile) return;
    setIsDetecting(true);
    setVisionError(null);

    const formData = new FormData();
    formData.append("file", visionFile);
    formData.append("confidence", visionConf.toString());

    try {
      const expTarget = experimentId || "default";
      const res = await fetch(`http://localhost:8000/api/models/${expTarget}/predict_vision`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setVisionResult(data);
      } else {
        const err = await res.json().catch(() => ({}));
        setVisionError(err.detail || `Inference error: ${res.status}`);
      }
    } catch (err) {
      setVisionError("Network error during vision inference: " + err.message);
    } finally {
      setIsDetecting(false);
    }
  };

  const TABS = [
    { id: "tabular", num: "01", label: "TABULAR ML", desc: "Structured data prediction", placeholder: "e.g., 'Predict customer churn probability' or 'Maximize AUC-ROC'", fileHint: "CSV · XLSX · PARQUET" },
    { id: "llm", num: "02", label: "LLM FINE-TUNING", desc: "Instruction tuning", placeholder: "e.g., 'Fine-tune a model to output strict JSON responses.'", fileHint: "JSONL" },
    { id: "rag", num: "03", label: "RAG Q&A", desc: "Document intelligence", placeholder: "e.g., 'Create an index to answer questions about these PDFs.'", fileHint: "PDF · TXT · MD" },
    { id: "vision", num: "04", label: "COMPUTER VISION", desc: "Image classification", placeholder: "e.g., 'Detect cars and pedestrians in images.'", fileHint: "ZIP · PNG · JPG" }
  ];

  return (
    <div className="relative z-10 max-w-[1440px] mx-auto px-margin-desktop py-margin-desktop min-h-[calc(100vh-48px)] flex flex-col md:flex-row gap-16">

      <div className="w-full flex flex-col gap-12 mx-auto max-w-[1440px]">
        {/* Hero Header */}
        <header className="flex flex-col gap-4">
          <span className="text-label-caps font-label-caps text-on-surface-variant">MODEL BUILDER / 01</span>
          <div>
            <h1 className="text-display-lg font-display-lg text-on-background mb-2">Build your model.</h1>
            <p className="text-body-md font-body-md text-on-surface-variant max-w-xl">
              Select your pipeline and provide your training data. We construct and tune the model architecture automatically.
            </p>
          </div>
        </header>

        {/* Pipeline Selector */}
        <section className="flex flex-col gap-6">
          <h2 className="text-label-caps font-label-caps text-on-surface-variant border-b border-[#1A1A1A] pb-2">PIPELINE SELECTION</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {TABS.map(tab => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex flex-col gap-2 p-4 bg-[#0A0A0A] border text-left transition-colors relative group ${isActive ? "border-[#00E5FF]" : "border-[#1A1A1A] hover:border-[#333333]"}`}
                >
                  {isActive && (
                    <div className="absolute top-0 right-0 p-2 opacity-100">
                      <div className="w-2 h-2 bg-[#00E5FF]"></div>
                    </div>
                  )}
                  <span className={`text-label-caps font-label-caps transition-colors ${isActive ? "text-[#00E5FF]" : "text-on-surface-variant group-hover:text-on-background"}`}>
                    {tab.num} {tab.label}
                  </span>
                  <span className={`text-code-sm font-code-sm transition-colors ${isActive ? "text-on-surface-variant group-hover:text-on-background" : "text-[#595959]"}`}>
                    {tab.desc}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Builder Form Area */}
        <section className="bg-[#0A0A0A] border border-[#1A1A1A] flex flex-col">
          {/* Model Section */}
          <div className="p-6 border-b border-[#1A1A1A] flex flex-col md:flex-row gap-6 items-start md:items-center justify-between">
            <div className="w-48 flex-shrink-0">
              <label className="text-label-caps font-label-caps text-on-surface-variant">RECOMMENDED MODEL</label>
            </div>

            <div className="flex-grow flex flex-col sm:flex-row items-stretch sm:items-center gap-4 w-full">
              <div className="flex-grow flex items-center justify-between border-b border-[#333333] pb-2 group hover:border-[#00E5FF] transition-colors cursor-pointer relative">
                <select
                  value={modelCandidate}
                  onChange={(e) => setModelCandidate(e.target.value)}
                  disabled={isBuilding}
                  className="w-full text-body-md font-body-md text-on-background appearance-none cursor-pointer outline-none border-none bg-transparent m-0 p-0 shadow-none focus:outline-none focus:ring-0 focus:border-none focus:shadow-none"
                  style={{ borderBottom: "none" }}
                >
                  {activeTab === "tabular" && (
                    <optgroup label="Tabular / AutoGluon">
                      <option value="autogluon_best">AutoGluon Best</option>
                    </optgroup>
                  )}
                  {activeTab === "llm" && (
                    <>
                      <optgroup label="LLM / Unsloth Llama Series">
                        <option value="unsloth/Llama-3.2-1B-Instruct-bnb-4bit">Llama 3.2 1B Instruct</option>
                        <option value="unsloth/Llama-3.2-3B-Instruct-bnb-4bit">Llama 3.2 3B Instruct</option>
                        <option value="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">Llama 3.1 8B Instruct</option>
                        <option value="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit">Llama 3.1 70B Instruct</option>
                        <option value="unsloth/Llama-3.3-70B-Instruct-bnb-4bit">Llama 3.3 70B Instruct</option>
                      </optgroup>
                      <optgroup label="LLM / Unsloth DeepSeek Series">
                        <option value="unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit">DeepSeek R1 (Distill Llama 8B)</option>
                        <option value="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit">DeepSeek R1 (Distill Qwen 1.5B)</option>
                        <option value="unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit">DeepSeek R1 (Distill Qwen 7B)</option>
                        <option value="unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit">DeepSeek R1 (Distill Qwen 14B)</option>
                        <option value="unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit">DeepSeek R1 (Distill Qwen 32B)</option>
                      </optgroup>
                      <optgroup label="LLM / Unsloth Qwen Series">
                        <option value="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit">Qwen 2.5 0.5B Instruct</option>
                        <option value="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit">Qwen 2.5 1.5B Instruct</option>
                        <option value="unsloth/Qwen2.5-3B-Instruct-bnb-4bit">Qwen 2.5 3B Instruct</option>
                        <option value="unsloth/Qwen2.5-7B-Instruct-bnb-4bit">Qwen 2.5 7B Instruct</option>
                        <option value="unsloth/Qwen2.5-14B-Instruct-bnb-4bit">Qwen 2.5 14B Instruct</option>
                        <option value="unsloth/Qwen2.5-32B-Instruct-bnb-4bit">Qwen 2.5 32B Instruct</option>
                        <option value="unsloth/Qwen2.5-72B-Instruct-bnb-4bit">Qwen 2.5 72B Instruct</option>
                      </optgroup>
                      <optgroup label="LLM / Unsloth Coder Series">
                        <option value="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit">Qwen 2.5 Coder 1.5B Instruct</option>
                        <option value="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit">Qwen 2.5 Coder 7B Instruct</option>
                        <option value="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit">Qwen 2.5 Coder 32B Instruct</option>
                        <option value="unsloth/Llama-3-8B-Instruct-Coder-bnb-4bit">Llama 3 8B Coder</option>
                      </optgroup>
                      <optgroup label="LLM / Unsloth Mistral & Gemma & Phi">
                        <option value="unsloth/mistral-7b-instruct-v0.3-bnb-4bit">Mistral 7B Instruct v0.3</option>
                        <option value="unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit">Mistral Nemo 12B Instruct</option>
                        <option value="unsloth/gemma-2-2b-it-bnb-4bit">Gemma 2 2B Instruct</option>
                        <option value="unsloth/gemma-2-9b-it-bnb-4bit">Gemma 2 9B Instruct</option>
                        <option value="unsloth/gemma-2-27b-it-bnb-4bit">Gemma 2 27B Instruct</option>
                        <option value="unsloth/Phi-3.5-mini-instruct-bnb-4bit">Phi 3.5 Mini Instruct</option>
                      </optgroup>
                    </>
                  )}
                  {activeTab === "rag" && (
                    <optgroup label="RAG / Document Q&A">
                      <option value="rag_default">RAG Index (FAISS + Embeddings)</option>
                    </optgroup>
                  )}
                  {activeTab === "vision" && (
                    <optgroup label="Computer Vision / YOLOv8 & HF">
                      <option value="yolov8n">YOLOv8 Nano (Fastest, Edge Real-time)</option>
                      <option value="yolov8s">YOLOv8 Small (Balanced Speed/Accuracy)</option>
                      <option value="yolov8m">YOLOv8 Medium (High Accuracy Detection)</option>
                      <option value="autotrain_vision">HF AutoTrain (Vision Transformers)</option>
                    </optgroup>
                  )}
                  {importedModels.length > 0 && (
                    <optgroup label="Imported from Hugging Face Hub">
                      {importedModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.id} [{m.tasks?.[0] || m.pipeline_type || "custom"}]
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
                <span className="absolute right-0 top-1/2 -translate-y-1/2 text-label-caps font-label-caps text-on-surface-variant group-hover:text-[#00E5FF] transition-colors pointer-events-none bg-[#0A0A0A] pl-2">
                  [ SELECT ↓ ]
                </span>
              </div>

              <button
                type="button"
                onClick={() => {
                  setHfQuery(activeTab === "vision" ? "yolo" : activeTab === "tabular" ? "xgboost" : "llama");
                  setHfTask(activeTab === "vision" ? "object-detection" : activeTab === "llm" ? "text-generation" : "");
                  setShowHfModal(true);
                }}
                disabled={isBuilding}
                className="shrink-0 text-label-caps font-label-caps px-3.5 py-2 border border-[#333333] hover:border-[#00E5FF] hover:text-[#00E5FF] text-on-surface-variant transition-colors flex items-center justify-center gap-1.5 bg-[#131313]"
                title="Search and import models directly from Hugging Face Hub"
              >
                <Sparkles size={13} className="text-[#00E5FF]" />
                [ SEARCH HF HUB ]
              </button>
            </div>
          </div>

          {/* Dataset Section */}
          <div className="p-6 border-b border-[#1A1A1A] flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <label className="text-label-caps font-label-caps text-on-surface-variant">TRAINING DATASET</label>
              <span className="text-code-sm font-code-sm text-[#595959]">{TABS.find(t => t.id === activeTab).fileHint}</span>
            </div>

            {!manifest ? (
              <div
                className="border border-dashed border-[#333333] hover:border-[#00E5FF] transition-colors bg-[#131313] p-12 flex flex-col items-center justify-center gap-4 cursor-pointer group"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <UploadCloud size={38} className="text-[#595959] group-hover:text-[#00E5FF] transition-colors" />
                <div className="text-center flex flex-col gap-1">
                  <span className="text-body-md font-body-md text-on-background">
                    {isUploading ? "Uploading..." : "Drop your dataset here"}
                  </span>
                  <span className="text-code-sm font-code-sm text-[#595959]">or click to browse local files</span>
                </div>
                <button className="mt-4 px-6 py-2 border border-[#1A1A1A] text-label-caps font-label-caps text-on-background hover:border-[#00E5FF] transition-colors">
                  [ BROWSE FILES ]
                </button>
                <input
                  type="file"
                  multiple
                  hidden
                  ref={fileInputRef}
                  onChange={(e) => handleFiles(Array.from(e.target.files))}
                />
              </div>
            ) : (
              <div className="border border-[#1A1A1A] bg-[#131313] w-full">
                <table className="w-full text-left">
                  <thead className="border-b border-[#1A1A1A] bg-[#0A0A0A]">
                    <tr>
                      <th className="p-3 text-label-caps font-label-caps text-[#595959]">FILENAME</th>
                      <th className="p-3 text-label-caps font-label-caps text-[#595959]">TYPE</th>
                      <th className="p-3 text-label-caps font-label-caps text-[#595959]">SIZE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {manifest.map((item, i) => (
                      <tr key={i} className="border-b border-[#1A1A1A] last:border-b-0 hover:bg-[#1f1f1f] transition-colors">
                        <td className="p-3 text-code-sm font-code-sm text-on-background flex items-center gap-2">
                          <CheckCircle2 size={14} className="text-[#00E5FF]" />
                          {item.filename}
                        </td>
                        <td className="p-3">
                          <span className="text-label-caps font-label-caps px-2 py-1 bg-[#1A1A1A] text-on-surface-variant border border-[#333333]">
                            {item.file_type}
                          </span>
                        </td>
                        <td className="p-3 text-code-sm font-code-sm text-[#595959]">
                          {(item.size_bytes / 1024).toFixed(1)} KB
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

        </section>

        {/* Action Area */}
        <div className="flex justify-end pt-4">
          <button
            className="bg-[#00E5FF] text-black px-8 py-3 text-label-caps font-label-caps font-bold hover:bg-white hover:text-black transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleBuild}
            disabled={!manifest || isBuilding}
          >
            {isBuilding ? "BUILDING..." : "BUILD MODEL"} <ArrowRight size={16} />
          </button>
        </div>

        {/* Status Logs (Brutalist Terminal) */}
        {statusLogs.length > 0 && (
          <section className="flex flex-col gap-4 mt-8">
            <h2 className="text-label-caps font-label-caps text-on-surface-variant border-b border-[#1A1A1A] pb-2">PIPELINE EXECUTION LOG</h2>
            <div className="bg-[#0e0e0e] border border-[#1A1A1A] p-4 h-64 overflow-y-auto">
              {statusLogs.map((log, idx) => (
                <div key={idx} className="flex gap-4 mb-2 text-code-sm font-code-sm">
                  <span className="text-[#595959] shrink-0">{fmtTime(log.updated_at)}</span>
                  <span className={`shrink-0 w-24 uppercase ${log.stage === "failed" ? "text-error" : "text-on-surface-variant"}`}>
                    [{log.stage}]
                  </span>
                  <span className={log.stage === "completed" ? "text-[#00E5FF]" : "text-on-background"}>
                    {log.message}
                  </span>
                  {log.pct > 0 && <span className="text-[#00E5FF] shrink-0">{log.pct}%</span>}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </section>
        )}

        {/* Results & Chat */}
        {finalMetrics && finalMetrics.leaderboard && (
          <section className="flex flex-col gap-4 mt-8">
            <div className="flex justify-between items-end border-b border-[#1A1A1A] pb-2">
              <h2 className="text-label-caps font-label-caps text-on-surface-variant">TRAINING RESULTS</h2>
              <a
                href={`http://localhost:8000/api/experiments/${experimentId}/download`}
                className="text-label-caps font-label-caps text-[#00E5FF] hover:text-white transition-colors flex items-center gap-1"
                download
              >
                [ DOWNLOAD MODEL ]
              </a>
            </div>

            <div className="border border-[#1A1A1A] bg-[#0e0e0e] w-full">
              <table className="w-full text-left">
                <thead className="border-b border-[#1A1A1A] bg-[#131313]">
                  <tr>
                    <th className="p-3 text-label-caps font-label-caps text-[#595959]">MODEL</th>
                    <th className="p-3 text-label-caps font-label-caps text-[#595959]">SCORE</th>
                    <th className="p-3 text-label-caps font-label-caps text-[#595959]">FIT TIME (S)</th>
                    <th className="p-3 text-label-caps font-label-caps text-[#595959]">PRED TIME (S)</th>
                  </tr>
                </thead>
                <tbody>
                  {finalMetrics.leaderboard
                    .sort((a, b) => b.score - a.score)
                    .map((row, i) => (
                      <tr key={i} className={`border-b border-[#1A1A1A] last:border-b-0 ${row.is_best ? "bg-[#00363d]/30" : "hover:bg-[#1f1f1f]"}`}>
                        <td className="p-3 text-code-sm font-code-sm text-on-background flex items-center gap-2">
                          {row.is_best && <div className="w-2 h-2 bg-[#00E5FF]"></div>}
                          {row.model_name}
                        </td>
                        <td className="p-3 text-code-sm font-code-sm text-[#00E5FF]">{row.score.toFixed(4)}</td>
                        <td className="p-3 text-code-sm font-code-sm text-[#595959]">{row.fit_time.toFixed(2)}</td>
                        <td className="p-3 text-code-sm font-code-sm text-[#595959]">{row.pred_time.toFixed(4)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Inference / Chat UI */}
        {(experimentBackend === "unsloth" || (finalMetrics && finalMetrics.retrieval_accuracy !== undefined)) && (
          <section className="flex flex-col gap-4 mt-8 mb-24">
            <div className="flex items-center justify-between border-b border-[#1A1A1A] pb-2">
              <h2 className="text-label-caps font-label-caps text-on-surface-variant">
                {experimentBackend === "unsloth" ? "INFERENCE ENDPOINT" : "RAG ENDPOINT"}
              </h2>
              {experimentId && !isBuilding && (
                <a
                  href={`http://localhost:8000/api/experiments/${experimentId}/download`}
                  download
                  className="px-4 py-1.5 bg-[#00E5FF]/10 border border-[#00E5FF]/40 text-[#00E5FF] hover:bg-[#00E5FF] hover:text-black font-mono text-xs font-bold transition-all flex items-center gap-2"
                >
                  <span>⬇</span>
                  <span>DOWNLOAD MODEL (.ZIP)</span>
                </a>
              )}
            </div>

            <div className="bg-[#0e0e0e] border border-[#1A1A1A] p-6 flex flex-col min-h-[400px]">
              <div className="flex-1 overflow-y-auto mb-4 flex flex-col gap-6">
                {ragChat.length === 0 && (
                  isBuilding ? (
                    <div className="text-code-sm font-code-sm text-[#00E5FF] italic text-center mt-12 flex flex-col items-center gap-2">
                      <div className="flex items-center gap-2 font-mono">
                        <span className="inline-block w-2 h-2 rounded-full bg-[#00E5FF] animate-ping" />
                        <span>MODEL TRAINING IN PROGRESS...</span>
                      </div>
                      <span className="text-[#595959] text-xs">
                        The inference endpoint will activate automatically once training completes.
                      </span>
                    </div>
                  ) : (
                    <div className="text-code-sm font-code-sm text-[#595959] italic text-center mt-10">
                      Endpoint active. Awaiting input...
                    </div>
                  )
                )}

                {ragChat.map((msg, i) => (
                  <div key={i} className="flex flex-col gap-1">
                    <span className={`text-label-caps font-label-caps ${msg.role === "user" ? "text-on-surface-variant" : "text-[#00E5FF]"}`}>
                      {msg.role === "user" ? "INPUT_QUERY" : "MODEL_OUTPUT"}
                    </span>
                    <div className="text-body-md font-body-md text-on-background whitespace-pre-wrap">
                      {msg.content}
                    </div>
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-2 pl-4 border-l border-[#333333] flex flex-col gap-1">
                        <span className="text-label-caps font-label-caps text-[#595959]">CONTEXT CITATIONS</span>
                        <ul className="list-none m-0 p-0 text-code-sm font-code-sm text-[#595959]">
                          {msg.citations.map((c, idx) => (
                            <li key={idx}>
                              [{idx + 1}] {c.metadata.source} {c.metadata.page ? `(Page ${c.metadata.page})` : ""}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}

                {isQuerying && (
                  <div className="text-code-sm font-code-sm text-[#00E5FF] animate-pulse">
                    Processing request...
                  </div>
                )}
              </div>

              <form onSubmit={handleChatSubmit} className="flex gap-4 border-t border-[#1A1A1A] pt-4">
                <input
                  type="text"
                  value={ragQuery}
                  onChange={e => setRagQuery(e.target.value)}
                  placeholder={isBuilding ? "Model training in progress..." : "Enter query..."}
                  disabled={isQuerying || isBuilding}
                  className="flex-1 bg-transparent border-none border-b border-[#333333] text-body-md font-body-md text-on-background p-2 focus:border-[#00E5FF] focus:outline-none transition-colors disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={isQuerying || isBuilding || !ragQuery.trim()}
                  className="bg-transparent border border-[#333333] text-on-surface-variant px-6 py-2 text-label-caps font-label-caps hover:border-[#00E5FF] hover:text-[#00E5FF] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  [ SEND ]
                </button>
              </form>
            </div>
          </section>
        )}

        {/* Computer Vision Section */}
        {(experimentBackend === "ultralytics" || activeTab === "vision") && (
          <section className="flex flex-col gap-6 mt-8 mb-24">
            {/* Vision Sub-Mode Switcher */}
            <div className="flex border-b border-[#1A1A1A] gap-6 flex-wrap">
              <button
                type="button"
                onClick={() => setVisionSubMode("classifier")}
                className={`pb-3 text-label-caps font-label-caps border-b-2 transition-colors flex items-center gap-2 ${visionSubMode === "classifier" || visionSubMode === "teachable"
                  ? "border-[#00E5FF] text-[#00E5FF]"
                  : "border-transparent text-on-surface-variant hover:text-on-background"
                  }`}
              >
                <Sparkles size={14} className={visionSubMode === "classifier" || visionSubMode === "teachable" ? "text-[#00E5FF]" : "text-[#595959]"} />
                CUSTOM VISION CLASSIFIER (IN-HOUSE)
              </button>
              <button
                type="button"
                onClick={() => setVisionSubMode("detection")}
                className={`pb-3 text-label-caps font-label-caps border-b-2 transition-colors flex items-center gap-2 ${visionSubMode === "detection"
                  ? "border-[#00E5FF] text-[#00E5FF]"
                  : "border-transparent text-on-surface-variant hover:text-on-background"
                  }`}
              >
                <ImageIcon size={14} className={visionSubMode === "detection" ? "text-[#00E5FF]" : "text-[#595959]"} />
                OBJECT DETECTION (YOLOV8 STUDIO)
              </button>
            </div>

            {visionSubMode === "classifier" || visionSubMode === "teachable" ? (
              <CustomVisionStudio />
            ) : (
              <div className="flex flex-col gap-4">
                <div className="flex justify-between items-end border-b border-[#1A1A1A] pb-2 flex-wrap gap-2">
                  <div>
                    <h2 className="text-label-caps font-label-caps text-on-surface-variant">
                      COMPUTER VISION INFERENCE STUDIO (YOLOV8)
                    </h2>
                    <p className="text-code-sm font-code-sm text-[#595959] mt-1">
                      Upload test images or run sample detection with confidence threshold control.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleVisionSample}
                    disabled={isDetecting}
                    className="text-label-caps font-label-caps px-3 py-1.5 border border-[#333333] hover:border-[#00E5FF] text-on-surface-variant hover:text-[#00E5FF] transition-colors flex items-center gap-1.5 bg-[#131313]"
                  >
                    <Sparkles size={13} className="text-[#00E5FF]" />
                    [ TRY SAMPLE TEST IMAGE ]
                  </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 bg-[#0A0A0A] border border-[#1A1A1A] p-6">
                  {/* Controls Column */}
                  <div className="flex flex-col gap-4">
                    <div
                      className="border border-dashed border-[#333333] hover:border-[#00E5FF] transition-colors bg-[#131313] p-6 flex flex-col items-center justify-center gap-3 cursor-pointer group min-h-[220px]"
                      onClick={() => visionInputRef.current?.click()}
                    >
                      {visionPreview ? (
                        <div className="text-center">
                          <img
                            src={visionPreview}
                            alt="Test image"
                            className="max-h-48 max-w-full rounded border border-[#1A1A1A] object-contain mb-2 mx-auto"
                          />
                          <span className="text-code-sm font-code-sm text-[#595959] group-hover:text-[#00E5FF] transition-colors">
                            Click to change image
                          </span>
                        </div>
                      ) : (
                        <>
                          <UploadCloud size={32} className="text-[#595959] group-hover:text-[#00E5FF] transition-colors" />
                          <div className="text-center flex flex-col gap-1">
                            <span className="text-body-md font-body-md text-on-background">
                              Drop test image here or browse
                            </span>
                            <span className="text-code-sm font-code-sm text-[#595959]">Supports JPG, PNG, WEBP</span>
                          </div>
                        </>
                      )}
                      <input
                        type="file"
                        accept="image/*"
                        hidden
                        ref={visionInputRef}
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            handleVisionFileSelect(e.target.files[0]);
                          }
                        }}
                      />
                    </div>

                    {/* Confidence Threshold & Action */}
                    <div className="flex flex-col gap-2 p-4 bg-[#131313] border border-[#1A1A1A]">
                      <div className="flex justify-between text-label-caps font-label-caps">
                        <span className="text-on-surface-variant flex items-center gap-1.5">
                          <Sliders size={13} className="text-[#00E5FF]" /> CONFIDENCE THRESHOLD
                        </span>
                        <span className="text-[#00E5FF] font-code-sm font-code-sm">{(visionConf * 100).toFixed(0)}%</span>
                      </div>
                      <input
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={visionConf}
                        onChange={(e) => setVisionConf(parseFloat(e.target.value))}
                        className="accent-[#00E5FF] cursor-pointer"
                      />
                    </div>

                    <button
                      type="button"
                      onClick={handleRunVisionInference}
                      disabled={isDetecting || !visionFile}
                      className="bg-[#00E5FF] text-black py-3 text-label-caps font-label-caps font-bold hover:bg-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {isDetecting ? "DETECTING OBJECTS..." : "[ RUN OBJECT DETECTION ]"}
                    </button>

                    {visionError && (
                      <div className="p-3 border border-red-800 bg-red-950/40 text-red-400 text-code-sm font-code-sm">
                        {visionError}
                      </div>
                    )}
                  </div>

                  {/* Output Column */}
                  <div className="flex flex-col bg-[#0e0e0e] border border-[#1A1A1A] p-4 min-h-[300px]">
                    <div className="flex justify-between items-center border-b border-[#1A1A1A] pb-3 mb-3">
                      <span className="text-label-caps font-label-caps text-on-surface-variant">DETECTION VISUALIZATION</span>
                      {visionResult && (
                        <div className="flex items-center gap-2">
                          <span className="text-label-caps font-label-caps px-2 py-0.5 bg-[#00363d] text-[#00E5FF] border border-[#00626e]">
                            {visionResult.count} {visionResult.count === 1 ? "OBJECT" : "OBJECTS"}
                          </span>
                          {visionResult.speed_ms?.inference && (
                            <span className="text-code-sm font-code-sm text-[#595959]">
                              {visionResult.speed_ms.inference.toFixed(1)}ms
                            </span>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="flex-1 flex items-center justify-center">
                      {visionResult?.annotated_image ? (
                        <img
                          src={visionResult.annotated_image}
                          alt="Detection Results"
                          className="max-h-72 max-w-full object-contain rounded border border-[#1A1A1A]"
                        />
                      ) : (
                        <div className="text-center text-code-sm font-code-sm text-[#595959] p-8">
                          {isDetecting ? "Running YOLOv8 inference..." : "Upload an image and run detection to view bounding boxes"}
                        </div>
                      )}
                    </div>

                    {visionResult?.detections && visionResult.detections.length > 0 && (
                      <div className="border-t border-[#1A1A1A] pt-3 mt-3 flex flex-col gap-2">
                        <span className="text-label-caps font-label-caps text-[#595959]">DETECTED LABELS:</span>
                        <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
                          {visionResult.detections.map((d, idx) => (
                            <span
                              key={idx}
                              className="px-2.5 py-1 bg-[#131313] border border-[#333333] text-code-sm font-code-sm text-on-background flex items-center gap-2"
                            >
                              <span className="text-[#00E5FF] font-semibold">{d.class}</span>
                              <span className="text-[#595959]">{(d.confidence * 100).toFixed(1)}%</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

      </div>

      {/* Hugging Face Search & Import Modal */}
      {showHfModal && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 backdrop-blur-sm">
          <div className="bg-[#0A0A0A] border border-[#1A1A1A] max-w-3xl w-full p-6 flex flex-col gap-4 shadow-2xl max-h-[85vh] overflow-hidden">
            <div className="flex justify-between items-start border-b border-[#1A1A1A] pb-3">
              <div>
                <h3 className="text-label-caps font-label-caps text-[#00E5FF]">HUGGING FACE MODEL HUB</h3>
                <p className="text-code-sm font-code-sm text-[#595959] mt-0.5">
                  Search 100,000+ community models and import directly into this pipeline.
                </p>
              </div>
              <button
                onClick={() => setShowHfModal(false)}
                className="text-label-caps font-label-caps text-on-surface-variant hover:text-white px-2 py-1 border border-[#333333]"
              >
                [ CLOSE ✕ ]
              </button>
            </div>

            {/* Search inputs */}
            <form onSubmit={handleHfSearch} className="flex flex-col sm:flex-row gap-3">
              <input
                type="text"
                value={hfQuery}
                onChange={(e) => setHfQuery(e.target.value)}
                placeholder="Search models (e.g., yolov8, phi-3, qwen, resnet)..."
                className="flex-1 bg-[#131313] border border-[#333333] focus:border-[#00E5FF] text-body-md font-body-md text-on-background px-3 py-2 outline-none"
              />
              <select
                value={hfTask}
                onChange={(e) => setHfTask(e.target.value)}
                className="bg-[#131313] border border-[#333333] text-body-md font-body-md text-on-background px-3 py-2 outline-none cursor-pointer"
              >
                <option value="">All Tasks</option>
                <option value="object-detection">Object Detection</option>
                <option value="image-classification">Image Classification</option>
                <option value="text-generation">Text Generation</option>
                <option value="feature-extraction">Feature Extraction / Embeddings</option>
                <option value="tabular-classification">Tabular Classification</option>
              </select>
              <button
                type="submit"
                disabled={hfSearching || !hfQuery.trim()}
                className="bg-[#00E5FF] text-black px-5 py-2 text-label-caps font-label-caps font-bold hover:bg-white transition-colors disabled:opacity-50"
              >
                {hfSearching ? "SEARCHING..." : "[ SEARCH ]"}
              </button>
            </form>

            {hfError && (
              <div className="p-2.5 border border-red-800 bg-red-950/40 text-red-400 text-code-sm font-code-sm">
                {hfError}
              </div>
            )}

            {/* Results */}
            <div className="flex-1 overflow-y-auto flex flex-col gap-2.5 pr-1">
              {hfResults.length === 0 && !hfSearching && (
                <div className="text-center text-code-sm font-code-sm text-[#595959] py-12">
                  Enter a keyword and click [ SEARCH ] to discover Hugging Face models.
                </div>
              )}
              {hfResults.map((m) => (
                <div
                  key={m.id}
                  className="bg-[#131313] border border-[#1A1A1A] hover:border-[#333333] p-3 flex justify-between items-center gap-4 transition-colors"
                >
                  <div className="flex flex-col gap-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-body-md font-body-md text-on-background font-semibold truncate">
                        {m.id}
                      </span>
                      {m.pipeline_tag && (
                        <span className="text-label-caps font-label-caps px-2 py-0.5 bg-[#1A1A1A] text-[#00E5FF] border border-[#333333]">
                          {m.pipeline_tag}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-code-sm font-code-sm text-[#595959]">
                      <span>By: {m.author}</span>
                      <span>⬇ {(m.downloads || 0).toLocaleString()}</span>
                      <span>★ {(m.likes || 0).toLocaleString()}</span>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => handleHfImport(m.id)}
                    disabled={hfImporting}
                    className="shrink-0 text-label-caps font-label-caps px-4 py-2 border border-[#00E5FF] text-[#00E5FF] hover:bg-[#00E5FF] hover:text-black transition-colors disabled:opacity-50"
                  >
                    {hfImporting ? "IMPORTING..." : "[ IMPORT ]"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
