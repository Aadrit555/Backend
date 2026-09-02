"use client";

import { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  CheckCircle2,
  DownloadCloud,
  Play,
  Sparkles,
  Search,
  X,
  Image as ImageIcon,
  Sliders,
  Table,
  Cpu,
  FileText,
  Camera,
  Layers,
  ArrowRight,
  Terminal,
} from "lucide-react";
import CustomVisionStudio from "../components/CustomVisionStudio";

export default function BeginnerPage() {
  const [projectId, setProjectId] = useState("proj_init");

  useEffect(() => {
    setProjectId(`proj_${Math.random().toString(36).substr(2, 9)}`);
  }, []);

  const [modelCandidate, setModelCandidate] = useState("autogluon_best");
  const [activeTab, setActiveTab] = useState("vision");
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

  // Time formatter
  const fmtTime = (iso) => {
    if (!iso) return "--:--:--.---";
    try {
      const d = new Date(iso);
      return d.toTimeString().split(" ")[0] + "." + String(d.getMilliseconds()).padStart(3, "0");
    } catch {
      return iso;
    }
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
      case "tabular":
        setModelCandidate("autogluon_best");
        break;
      case "llm":
        setModelCandidate("unsloth/Llama-3.2-1B-Instruct-bnb-4bit");
        break;
      case "rag":
        setModelCandidate("rag_default");
        break;
      case "vision":
        setModelCandidate("yolov8n");
        break;
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
          setStatusLogs((prev) => {
            const frontendEntries = prev.filter((e) => e._frontend);
            const backendEntries = data.log.map((e) => ({ ...e, _backend: true }));
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

      const exp = data.experiments?.find((e) => e.id === experimentId || true);
      if (exp) {
        setExperimentBackend(exp.backend);
        if (exp.metrics_json) {
          try {
            setFinalMetrics(JSON.parse(exp.metrics_json));
          } catch (e) {}
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFiles = async (newFiles) => {
    setFiles(newFiles);
    setIsUploading(true);

    const formData = new FormData();
    newFiles.forEach((file) => formData.append("files", file));

    try {
      const res = await fetch(`http://localhost:8000/api/projects/${projectId}/upload`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setManifest(data.manifest);
      } else {
        alert("Upload failed. Please try again.");
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
    setStatusLogs([
      {
        stage: "started",
        message: "Initiating build...",
        pct: 0,
        updated_at: new Date().toISOString(),
        _frontend: true,
      },
    ]);

    try {
      const res = await fetch("http://localhost:8000/api/expert_build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          pipeline_type: activeTab,
          expert_config: { model_candidates: [modelCandidate] },
        }),
      });
      if (res.ok) {
        setStatusLogs((prev) => [
          ...prev,
          {
            stage: "planning",
            message: "Build initialized. Starting execution...",
            pct: 0,
            updated_at: new Date().toISOString(),
            _frontend: true,
          },
        ]);
        pollForExperiment();
      } else {
        setStatusLogs((prev) => [
          ...prev,
          {
            stage: "failed",
            message: "API error initiating build.",
            pct: 0,
            updated_at: new Date().toISOString(),
            _frontend: true,
          },
        ]);
        setIsBuilding(false);
      }
    } catch (err) {
      setStatusLogs((prev) => [
        ...prev,
        {
          stage: "failed",
          message: "Network error.",
          pct: 0,
          updated_at: new Date().toISOString(),
          _frontend: true,
        },
      ]);
      setIsBuilding(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!ragQuery.trim()) return;

    setRagChat((prev) => [...prev, { role: "user", content: ragQuery }]);
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
        setRagChat((prev) => [
          ...prev,
          { role: "assistant", content: msgContent, citations: data.citations },
        ]);
      } else {
        const errorData = await res.json().catch(() => ({}));
        const errorDetail = errorData.detail || `HTTP Error ${res.status}`;
        setRagChat((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${errorDetail}` },
        ]);
      }
    } catch (err) {
      setRagChat((prev) => [
        ...prev,
        { role: "assistant", content: `Network Error: ${err.message}` },
      ]);
    } finally {
      setIsQuerying(false);
      setRagQuery("");
    }
  };

  const pollForExperiment = () => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > 120) {
        clearInterval(interval);
        setStatusLogs((prev) => [
          ...prev,
          {
            stage: "failed",
            message: "Timeout waiting for orchestrator.",
            pct: 0,
            updated_at: new Date().toISOString(),
            _frontend: true,
          },
        ]);
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
              try {
                errStr = JSON.parse(exp.config_json).error || errStr;
              } catch (e) {}
              setStatusLogs((prev) => [
                ...prev,
                {
                  stage: "failed",
                  message: `Orchestrator Failed: ${errStr}`,
                  pct: 0,
                  _frontend: true,
                },
              ]);
              setIsBuilding(false);
            } else {
              setExperimentId(exp.id);
              setExperimentBackend(exp.backend);
              setStatusLogs((prev) => [
                ...prev,
                {
                  stage: "execution",
                  message: `Experiment ${exp.id} created. Starting training...`,
                  pct: 5,
                  updated_at: new Date().toISOString(),
                  _frontend: true,
                },
              ]);
            }
          }
        }
      } catch (e) {}
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
      const res = await fetch(
        `http://localhost:8000/api/hf/search?query=${encodeURIComponent(hfQuery)}${taskParam}&limit=15`
      );
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
      setHfError("Network error during model import.");
    } finally {
      setHfImporting(false);
    }
  };

  // Vision Inference Studio Handlers
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

    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = "#334155";
    ctx.fillRect(0, 320, 640, 160);

    ctx.fillStyle = "#e2e8f0";
    for (let x = 40; x < 600; x += 100) {
      ctx.fillRect(x, 395, 50, 10);
    }

    ctx.fillStyle = "#ef4444";
    ctx.fillRect(180, 260, 200, 90);
    ctx.fillStyle = "#f87171";
    ctx.fillRect(220, 200, 120, 60);

    ctx.fillStyle = "#0f172a";
    ctx.beginPath();
    ctx.arc(210, 340, 32, 0, Math.PI * 2);
    ctx.arc(350, 340, 32, 0, Math.PI * 2);
    ctx.fill();

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
    {
      id: "vision",
      label: "Computer Vision",
      icon: Camera,
      fileHint: "WEBCAM · JPG · PNG · ZIP",
    },
    {
      id: "tabular",
      label: "Tabular AutoML",
      icon: Table,
      fileHint: "CSV · XLSX · PARQUET",
    },
    {
      id: "llm",
      label: "LLM Fine-Tuning",
      icon: Cpu,
      fileHint: "JSONL",
    },
    {
      id: "rag",
      label: "RAG Assistant",
      icon: FileText,
      fileHint: "PDF · TXT · MD",
    },
  ];

  return (
    <div className="relative z-10 max-w-[1360px] mx-auto px-5 py-5 flex flex-col gap-5 text-[#E0E0E0]">
      {/* Top Pipeline Bar */}
      <div className="flex items-center justify-between border-b border-[#222222] pb-3 flex-wrap gap-3">
        <div className="flex items-center gap-1.5 bg-[#0F0F0F] p-1 border border-[#222222] rounded-xs">
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            const Icon = tab.icon;

            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`px-3.5 py-1.5 text-xs font-mono transition-all flex items-center gap-2 rounded-xs ${
                  isActive
                    ? "bg-[#1C1C1C] text-[#00E5FF] font-bold border border-[#333333] shadow-[0_0_12px_rgba(0,229,255,0.1)]"
                    : "text-[#888888] hover:text-white hover:bg-[#161616] border border-transparent"
                }`}
              >
                <Icon size={14} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        <button
          type="button"
          onClick={() => {
            setHfQuery(
              activeTab === "vision"
                ? "yolo"
                : activeTab === "tabular"
                ? "xgboost"
                : "llama"
            );
            setHfTask(
              activeTab === "vision"
                ? "object-detection"
                : activeTab === "llm"
                ? "text-generation"
                : ""
            );
            setShowHfModal(true);
          }}
          disabled={isBuilding}
          className="px-3 py-1.5 text-xs font-mono bg-[#111111] hover:bg-[#1A1A1A] border border-[#2A2A2A] hover:border-[#00E5FF] text-[#AAAAAA] hover:text-[#00E5FF] transition-colors flex items-center gap-1.5"
        >
          <Search size={13} />
          Hugging Face Hub
        </button>
      </div>

      {/* =================================================================== */}
      {/* 1. COMPUTER VISION WORKSPACE                                        */}
      {/* =================================================================== */}
      {activeTab === "vision" && (
        <div className="flex flex-col gap-4">
          {/* Sub-mode Switcher */}
          <div className="flex items-center justify-between border-b border-[#1E1E1E] pb-2 flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setVisionSubMode("classifier")}
                className={`px-3 py-1 text-xs font-mono transition-colors border ${
                  visionSubMode === "classifier"
                    ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white font-bold"
                    : "border-[#252525] bg-[#121212] text-[#888888] hover:text-white"
                }`}
              >
                Custom Vision Classifier (In-House)
              </button>
              <button
                type="button"
                onClick={() => setVisionSubMode("detection")}
                className={`px-3 py-1 text-xs font-mono transition-colors border ${
                  visionSubMode === "detection"
                    ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white font-bold"
                    : "border-[#252525] bg-[#121212] text-[#888888] hover:text-white"
                }`}
              >
                Object Detection (YOLOv8)
              </button>
            </div>

            <span className="text-[11px] font-mono text-[#666666]">
              {visionSubMode === "classifier"
                ? "Sub-second PyTorch transfer learning on webcam or image data"
                : "Ultralytics YOLOv8 inference with bounding box rendering"}
            </span>
          </div>

          {/* Sub-mode Content */}
          {visionSubMode === "classifier" ? (
            <CustomVisionStudio />
          ) : (
            <div className="flex flex-col gap-4">
              <div className="flex justify-between items-center border-b border-[#1E1E1E] pb-2 flex-wrap gap-2">
                <span className="text-xs font-mono text-[#AAAAAA] uppercase tracking-wider">
                  YOLOv8 Object Detection Studio
                </span>
                <button
                  type="button"
                  onClick={handleVisionSample}
                  disabled={isDetecting}
                  className="text-xs font-mono px-3 py-1 border border-[#333333] hover:border-[#00E5FF] text-white hover:text-[#00E5FF] transition-colors flex items-center gap-1.5 bg-[#141414]"
                >
                  <Sparkles size={12} className="text-[#00E5FF]" />
                  Load Sample Scene
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 bg-[#0F0F0F] border border-[#222222] p-5">
                {/* Left Controls */}
                <div className="flex flex-col gap-3">
                  <div
                    className="border border-dashed border-[#333333] hover:border-[#00E5FF] transition-colors bg-[#141414] p-6 flex flex-col items-center justify-center gap-3 cursor-pointer group min-h-[220px]"
                    onClick={() => visionInputRef.current?.click()}
                  >
                    {visionPreview ? (
                      <div className="text-center">
                        <img
                          src={visionPreview}
                          alt="Test image"
                          className="max-h-44 max-w-full border border-[#2A2A2A] object-contain mb-2 mx-auto"
                        />
                        <span className="text-xs font-mono text-[#777777] group-hover:text-[#00E5FF] transition-colors">
                          Click to change image
                        </span>
                      </div>
                    ) : (
                      <>
                        <UploadCloud
                          size={28}
                          className="text-[#666666] group-hover:text-[#00E5FF] transition-colors"
                        />
                        <div className="text-center flex flex-col gap-0.5">
                          <span className="text-xs font-mono text-white">
                            Drop test image here or click to browse
                          </span>
                          <span className="text-[11px] font-mono text-[#666666]">
                            Supports JPG, PNG, WEBP
                          </span>
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

                  {/* Confidence Slider */}
                  <div className="flex flex-col gap-1.5 p-3 bg-[#141414] border border-[#222222]">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-[#888888] flex items-center gap-1.5">
                        <Sliders size={12} className="text-[#00E5FF]" /> Confidence Threshold
                      </span>
                      <span className="text-[#00E5FF] font-bold">
                        {(visionConf * 100).toFixed(0)}%
                      </span>
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
                    className="bg-[#00E5FF] text-black py-2.5 px-4 text-xs font-mono uppercase font-bold hover:bg-[#00cbe2] transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isDetecting ? "DETECTING OBJECTS..." : "RUN OBJECT DETECTION"}
                  </button>

                  {visionError && (
                    <div className="p-2 border border-red-800 bg-red-950/40 text-red-300 text-xs font-mono">
                      {visionError}
                    </div>
                  )}
                </div>

                {/* Right Results */}
                <div className="flex flex-col gap-3">
                  <div className="border border-[#222222] bg-black min-h-[280px] flex items-center justify-center relative overflow-hidden">
                    {visionResult?.annotated_image ? (
                      <img
                        src={`data:image/jpeg;base64,${visionResult.annotated_image}`}
                        alt="YOLOv8 Detection"
                        className="w-full h-full object-contain max-h-[350px]"
                      />
                    ) : (
                      <div className="text-center p-4 flex flex-col items-center gap-1.5">
                        <ImageIcon size={28} className="text-[#444444]" />
                        <span className="text-xs font-mono text-[#666666]">
                          Run detection to view bounding boxes
                        </span>
                      </div>
                    )}
                  </div>

                  {visionResult?.predictions && (
                    <div className="flex flex-wrap gap-1.5">
                      {visionResult.predictions.map((p, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 bg-[#181818] border border-[#2D2D2D] text-xs font-mono text-[#00E5FF]"
                        >
                          {p.name} ({(p.confidence * 100).toFixed(0)}%)
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* =================================================================== */}
      {/* 2. TABULAR, LLM, AND RAG WORKSPACE                                  */}
      {/* =================================================================== */}
      {activeTab !== "vision" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Left Column: Model & Data Setup (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            <div className="bg-[#0F0F0F] border border-[#222222] p-4 flex flex-col gap-4">
              {/* Model Candidate Selector */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-mono text-[#888888] uppercase">
                    Select Target Model
                  </label>
                  <span className="text-[10px] font-mono text-[#666666]">
                    {TABS.find((t) => t.id === activeTab)?.fileHint}
                  </span>
                </div>

                <select
                  value={modelCandidate}
                  onChange={(e) => setModelCandidate(e.target.value)}
                  disabled={isBuilding}
                  className="w-full bg-[#161616] border border-[#2D2D2D] focus:border-[#00E5FF] text-white px-3 py-2 text-xs font-mono outline-none cursor-pointer"
                >
                  {activeTab === "tabular" && (
                    <optgroup label="AutoGluon Tabular Models">
                      <option value="autogluon_best">AutoGluon Best (Automatic Ensemble)</option>
                    </optgroup>
                  )}
                  {activeTab === "llm" && (
                    <>
                      <optgroup label="Unsloth LLaMA 3 Series (4-bit QLoRA)">
                        <option value="unsloth/Llama-3.2-1B-Instruct-bnb-4bit">
                          LLaMA 3.2 1B Instruct (Fastest)
                        </option>
                        <option value="unsloth/Llama-3.2-3B-Instruct-bnb-4bit">
                          LLaMA 3.2 3B Instruct (Recommended for 6GB VRAM)
                        </option>
                        <option value="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">
                          LLaMA 3.1 8B Instruct
                        </option>
                      </optgroup>
                      <optgroup label="Unsloth Qwen 2.5 Series (4-bit QLoRA)">
                        <option value="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit">
                          Qwen 2.5 0.5B Instruct
                        </option>
                        <option value="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit">
                          Qwen 2.5 1.5B Instruct
                        </option>
                        <option value="unsloth/Qwen2.5-3B-Instruct-bnb-4bit">
                          Qwen 2.5 3B Instruct
                        </option>
                        <option value="unsloth/Qwen2.5-7B-Instruct-bnb-4bit">
                          Qwen 2.5 7B Instruct
                        </option>
                      </optgroup>
                    </>
                  )}
                  {activeTab === "rag" && (
                    <optgroup label="Vector Search & Retrieval">
                      <option value="rag_default">FAISS Index (all-MiniLM-L6-v2)</option>
                    </optgroup>
                  )}
                  {importedModels.length > 0 && (
                    <optgroup label="Hugging Face Hub Imports">
                      {importedModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.id} [{m.tasks?.[0] || m.pipeline_type || "custom"}]
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </div>

              {/* Training Dataset Upload */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-mono text-[#888888] uppercase">
                  Dataset Input
                </label>

                {!manifest ? (
                  <div
                    className="border border-dashed border-[#333333] hover:border-[#00E5FF] transition-colors bg-[#141414] p-8 flex flex-col items-center justify-center gap-2 cursor-pointer group"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <UploadCloud
                      size={24}
                      className="text-[#666666] group-hover:text-[#00E5FF] transition-colors"
                    />
                    <span className="text-xs font-mono text-white">
                      {isUploading ? "Uploading files..." : "Drag & drop files or click to browse"}
                    </span>
                    <span className="text-[11px] font-mono text-[#666666]">
                      {TABS.find((t) => t.id === activeTab)?.fileHint}
                    </span>
                    <input
                      type="file"
                      multiple
                      hidden
                      ref={fileInputRef}
                      onChange={(e) => handleFiles(Array.from(e.target.files))}
                    />
                  </div>
                ) : (
                  <div className="border border-[#222222] bg-[#141414] overflow-hidden">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="border-b border-[#222222] bg-[#0E0E0E] text-[#777777]">
                        <tr>
                          <th className="p-2.5">FILE</th>
                          <th className="p-2.5">TYPE</th>
                          <th className="p-2.5">SIZE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {manifest.map((item, i) => (
                          <tr
                            key={i}
                            className="border-b border-[#1C1C1C] last:border-b-0 hover:bg-[#1A1A1A] transition-colors"
                          >
                            <td className="p-2.5 text-white flex items-center gap-1.5">
                              <CheckCircle2 size={12} className="text-[#00E5FF]" />
                              {item.filename}
                            </td>
                            <td className="p-2.5 text-[#888888]">{item.file_type}</td>
                            <td className="p-2.5 text-[#666666]">
                              {(item.size_bytes / 1024).toFixed(1)} KB
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Action Button */}
              <button
                type="button"
                onClick={handleBuild}
                disabled={!manifest || isBuilding}
                className="w-full bg-[#00E5FF] hover:bg-[#00cbe2] text-black py-2.5 px-4 text-xs font-mono uppercase font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 cursor-pointer shadow-[0_0_15px_rgba(0,229,255,0.15)]"
              >
                {isBuilding ? "EXECUTING PIPELINE..." : "START MODEL BUILD"}
                <ArrowRight size={13} />
              </button>
            </div>
          </div>

          {/* Right Column: Terminal Logs & Results (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            {/* Terminal Log */}
            <div className="border border-[#222222] bg-[#0B0B0B] p-3 flex flex-col gap-2 min-h-[260px] max-h-[380px]">
              <div className="flex items-center justify-between border-b border-[#1E1E1E] pb-1.5">
                <span className="text-[11px] font-mono text-[#888888] flex items-center gap-1.5">
                  <Terminal size={12} className="text-[#00E5FF]" /> EXECUTION LOGS
                </span>
                <span className="text-[10px] font-mono text-[#555555]">
                  {statusLogs.length} events
                </span>
              </div>

              <div className="flex-1 overflow-y-auto flex flex-col gap-1 text-[11px] font-mono">
                {statusLogs.length === 0 ? (
                  <div className="text-center text-[#555555] py-12">
                    Logs will stream here during pipeline execution.
                  </div>
                ) : (
                  statusLogs.map((log, idx) => (
                    <div key={idx} className="flex gap-2">
                      <span className="text-[#555555] shrink-0">{fmtTime(log.updated_at)}</span>
                      <span
                        className={`shrink-0 ${
                          log.stage === "failed"
                            ? "text-red-400 font-bold"
                            : log.stage === "completed"
                            ? "text-[#00E5FF] font-bold"
                            : "text-[#888888]"
                        }`}
                      >
                        [{log.stage}]
                      </span>
                      <span className="text-white break-all">{log.message}</span>
                    </div>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </div>

            {/* Results Leaderboard */}
            {finalMetrics?.leaderboard && (
              <div className="border border-[#222222] bg-[#0F0F0F] p-3 flex flex-col gap-2">
                <div className="flex justify-between items-center border-b border-[#1E1E1E] pb-1.5">
                  <span className="text-xs font-mono text-[#AAAAAA] uppercase">
                    Leaderboard
                  </span>
                  <a
                    href={`http://localhost:8000/api/experiments/${experimentId}/download`}
                    className="text-xs font-mono text-[#00E5FF] hover:underline flex items-center gap-1"
                    download
                  >
                    <DownloadCloud size={12} /> Download
                  </a>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="border-b border-[#1E1E1E] text-[#666666]">
                      <tr>
                        <th className="p-1.5">MODEL</th>
                        <th className="p-1.5">SCORE</th>
                        <th className="p-1.5">FIT (S)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {finalMetrics.leaderboard.map((row, i) => (
                        <tr
                          key={i}
                          className={`border-b border-[#1C1C1C] ${
                            row.is_best ? "text-[#00E5FF] font-bold" : "text-[#888888]"
                          }`}
                        >
                          <td className="p-1.5">{row.model_name}</td>
                          <td className="p-1.5">{row.score.toFixed(4)}</td>
                          <td className="p-1.5">{row.fit_time.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Interactive Chat / Inference */}
            {(experimentBackend === "unsloth" || finalMetrics?.retrieval_accuracy !== undefined) && (
              <div className="border border-[#222222] bg-[#0F0F0F] p-3 flex flex-col gap-2">
                <span className="text-xs font-mono text-[#AAAAAA] uppercase">
                  {experimentBackend === "unsloth" ? "Model Chat" : "RAG Assistant"}
                </span>

                <div className="min-h-[160px] max-h-[220px] overflow-y-auto flex flex-col gap-2 p-2 bg-black border border-[#1E1E1E] text-xs font-mono">
                  {ragChat.map((msg, i) => (
                    <div key={i} className="flex flex-col gap-0.5">
                      <span
                        className={`text-[10px] font-bold ${
                          msg.role === "user" ? "text-[#777777]" : "text-[#00E5FF]"
                        }`}
                      >
                        {msg.role === "user" ? "YOU" : "MODEL"}
                      </span>
                      <div className="text-white whitespace-pre-wrap">{msg.content}</div>
                    </div>
                  ))}
                  {isQuerying && (
                    <div className="text-[#00E5FF] animate-pulse">Generating response...</div>
                  )}
                </div>

                <form onSubmit={handleChatSubmit} className="flex gap-2">
                  <input
                    type="text"
                    value={ragQuery}
                    onChange={(e) => setRagQuery(e.target.value)}
                    placeholder="Ask model a question..."
                    disabled={isQuerying}
                    className="flex-1 bg-[#141414] border border-[#2D2D2D] focus:border-[#00E5FF] text-white px-2.5 py-1.5 text-xs font-mono outline-none"
                  />
                  <button
                    type="submit"
                    disabled={isQuerying || !ragQuery.trim()}
                    className="px-3 py-1.5 bg-[#00E5FF] text-black text-xs font-mono font-bold hover:bg-[#00cbe2] transition-colors disabled:opacity-40"
                  >
                    SEND
                  </button>
                </form>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Hugging Face Search & Import Modal */}
      {showHfModal && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 backdrop-blur-sm">
          <div className="bg-[#0F0F0F] border border-[#2A2A2A] max-w-2xl w-full p-5 flex flex-col gap-4 shadow-2xl max-h-[85vh] overflow-hidden">
            <div className="flex justify-between items-center border-b border-[#1E1E1E] pb-2.5">
              <div className="flex items-center gap-2">
                <Search size={15} className="text-[#00E5FF]" />
                <h3 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                  Hugging Face Model Hub
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setShowHfModal(false)}
                className="text-[#777777] hover:text-white p-1 transition-colors"
              >
                <X size={15} />
              </button>
            </div>

            {/* Search Input Form */}
            <form onSubmit={handleHfSearch} className="flex gap-2">
              <input
                type="text"
                value={hfQuery}
                onChange={(e) => setHfQuery(e.target.value)}
                placeholder="Search models (e.g. yolov8, phi-3, qwen, resnet)..."
                className="flex-1 bg-[#161616] border border-[#2D2D2D] focus:border-[#00E5FF] text-white px-3 py-1.5 text-xs font-mono outline-none"
              />
              <button
                type="submit"
                disabled={hfSearching || !hfQuery.trim()}
                className="bg-[#00E5FF] text-black px-4 py-1.5 text-xs font-mono font-bold hover:bg-[#00cbe2] transition-colors disabled:opacity-50"
              >
                {hfSearching ? "Searching..." : "Search"}
              </button>
            </form>

            {hfError && (
              <div className="p-2 border border-red-800 bg-red-950/40 text-red-300 text-xs font-mono">
                {hfError}
              </div>
            )}

            {/* Results */}
            <div className="flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
              {hfResults.length === 0 && !hfSearching && (
                <div className="text-center text-xs font-mono text-[#666666] py-10">
                  Enter keywords to search models on Hugging Face Hub.
                </div>
              )}
              {hfResults.map((m) => (
                <div
                  key={m.id}
                  className="bg-[#141414] border border-[#222222] hover:border-[#333333] p-2.5 flex justify-between items-center gap-3 transition-colors"
                >
                  <div className="flex flex-col gap-0.5 min-w-0">
                    <span className="text-xs font-mono text-white font-bold truncate">
                      {m.id}
                    </span>
                    <div className="flex items-center gap-3 text-[10px] font-mono text-[#666666]">
                      <span>by {m.author}</span>
                      <span>⬇ {(m.downloads || 0).toLocaleString()}</span>
                      <span>★ {(m.likes || 0).toLocaleString()}</span>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => handleHfImport(m.id)}
                    disabled={hfImporting}
                    className="shrink-0 text-xs font-mono px-3 py-1 border border-[#00E5FF] text-[#00E5FF] hover:bg-[#00E5FF] hover:text-black transition-colors disabled:opacity-50"
                  >
                    {hfImporting ? "Importing..." : "Import"}
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
