"use client";

import { useState, useRef, useEffect } from "react";
import { UploadCloud, CheckCircle2, AlertCircle, Terminal, Play, Cpu, Server, DownloadCloud } from "lucide-react";
import Link from "next/link";

export default function BeginnerPage() {
  const [projectId] = useState(`proj_${Math.random().toString(36).substr(2, 9)}`);
  const [modelCandidate, setModelCandidate] = useState("autogluon_best");
  const [activeTab, setActiveTab] = useState("tabular");
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
  
  const fileInputRef = useRef(null);
  const logsEndRef = useRef(null);

  // Poll for status
  // Helper to format an ISO timestamp to HH:MM:SS.mmm
  const fmtTime = (iso) => {
    if (!iso) return "--:--:--.---";
    try {
      const d = new Date(iso);
      return d.toTimeString().split(' ')[0] + '.' + String(d.getMilliseconds()).padStart(3, '0');
    } catch { return iso; }
  };

  useEffect(() => {
    switch (activeTab) {
      case "tabular": setModelCandidate("autogluon_best"); break;
      case "llm": setModelCandidate("unsloth/Llama-3.2-1B-Instruct-bnb-4bit"); break;
      case "rag": setModelCandidate("rag_default"); break;
      case "vision": setModelCandidate("yolov8"); break;
    }
  }, [activeTab]);

  useEffect(() => {
    if (!experimentId) return;
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/experiments/${experimentId}/status`);
        if (!res.ok) return;
        const data = await res.json();
        
        // Backend returns { ...current, log: [...entries] }
        // Merge backend log with our frontend-only entries (started, planning, execution)
        if (data.log && data.log.length > 0) {
          setStatusLogs(prev => {
            // Keep frontend-only entries (ones without updated_at from backend training)
            const frontendEntries = prev.filter(e => e._frontend);
            // Backend entries already have real timestamps
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
    e.currentTarget.classList.add("drag-active");
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("drag-active");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("drag-active");
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
                try { errStr = JSON.parse(exp.config_json).error || errStr; } catch(e) {}
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

  const TABS = [
    { id: "tabular", label: "Tabular ML", placeholder: "e.g. 'Predict whether a machine will fail based on the sensor readings.'", fileHint: "Upload CSV, Excel, or Parquet files" },
    { id: "llm", label: "LLM Fine-Tuning", placeholder: "e.g. 'Fine-tune a model to output strict JSON responses.'", fileHint: "Upload ShareGPT .jsonl files" },
    { id: "rag", label: "RAG Q&A", placeholder: "e.g. 'Create an index to answer questions about these PDFs.'", fileHint: "Upload PDF or TXT documents" },
    { id: "vision", label: "Computer Vision", placeholder: "e.g. 'Detect cars and pedestrians in images.'", fileHint: "Upload Images or ZIP archives" }
  ];

  return (
    <main>
      <h1>Unified Model Builder</h1>
      <p>Select your pipeline, provide your data, and define your objective. We construct the model.</p>

      <div style={{ display: "flex", gap: "10px", marginBottom: "2rem", borderBottom: "1px solid #333", paddingBottom: "10px" }}>
        {TABS.map(tab => (
          <button 
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{ 
              padding: "0.5rem 1rem", 
              background: activeTab === tab.id ? "#3b82f6" : "transparent",
              color: activeTab === tab.id ? "#fff" : "#aaa",
              border: "1px solid",
              borderColor: activeTab === tab.id ? "#3b82f6" : "#444",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <section className="card">
        <h2>Model Selection</h2>
        <select 
          value={modelCandidate} 
          onChange={(e) => setModelCandidate(e.target.value)}
          disabled={isBuilding}
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
            <optgroup label="Computer Vision">
              <option value="yolov8">YOLOv8 Default</option>
            </optgroup>
          )}
        </select>
      </section>

      <section className="card">
        <h2>Data Ingestion</h2>
        {!manifest ? (
          <div 
            className="dropzone"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadCloud size={32} style={{ marginBottom: "1rem", opacity: 0.7 }} />
            <div>
              {isUploading ? "Uploading..." : "Click or drag & drop files here"}
            </div>
            <div style={{ fontSize: "0.8rem", color: "#888", marginTop: "0.5rem" }}>
              {TABS.find(t => t.id === activeTab).fileHint}
            </div>
            <input 
              type="file" 
              multiple 
              hidden 
              ref={fileInputRef} 
              onChange={(e) => handleFiles(Array.from(e.target.files))}
            />
          </div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Type</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody>
                {manifest.map((item, i) => (
                  <tr key={i}>
                    <td className="data-mono">
                      <CheckCircle2 size={14} className="success" style={{ display: "inline", verticalAlign: "middle", marginRight: "6px" }}/>
                      {item.filename}
                    </td>
                    <td><span className="badge">{item.file_type}</span></td>
                    <td className="data-mono">{(item.size_bytes / 1024).toFixed(1)} KB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <button 
          className="primary" 
          onClick={handleBuild} 
          disabled={!manifest || isBuilding}
        >
          <Play size={16} style={{ marginRight: "8px" }}/> 
          {isBuilding ? "BUILDING..." : "BUILD MY MODEL"}
        </button>
      </div>

      {statusLogs.length > 0 && (
        <section className="card" style={{ marginTop: "2rem" }}>
          <h2>Pipeline Execution Log</h2>
          <div className="terminal">
            {statusLogs.map((log, idx) => (
              <div key={idx} className="line">
                <span style={{ color: "#555", marginRight: "1rem" }}>{fmtTime(log.updated_at)}</span>
                <span style={{ color: log.stage === "failed" ? "#f87171" : "#a1a1aa", textTransform: "uppercase", width: "100px", display: "inline-block" }}>
                  [{log.stage}]
                </span>
                <span className={log.stage === "completed" ? "success" : ""}>{log.message}</span>
                {log.pct > 0 && <span style={{ marginLeft: "1rem", color: "#60a5fa" }}>{log.pct}%</span>}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </section>
      )}

      {finalMetrics && finalMetrics.leaderboard && (
        <section className="card" style={{ marginTop: "2rem", marginBottom: "4rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h2>Training Results</h2>
            <a 
              href={`http://localhost:8000/api/experiments/${experimentId}/download`}
              className="primary"
              style={{ padding: "0.6rem 1.2rem", textDecoration: "none", display: "inline-flex", alignItems: "center", fontSize: "0.85rem", fontWeight: "600", borderRadius: "4px" }}
              download
            >
              <DownloadCloud size={16} style={{ marginRight: "8px" }} />
              Download Model
            </a>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Score</th>
                  <th>Fit Time (s)</th>
                  <th>Pred Time (s)</th>
                </tr>
              </thead>
              <tbody>
                {finalMetrics.leaderboard
                  .sort((a, b) => b.score - a.score)
                  .map((row, i) => (
                  <tr key={i} style={row.is_best ? { backgroundColor: "rgba(96, 165, 250, 0.1)" } : {}}>
                    <td>
                      <span style={{ fontWeight: row.is_best ? 600 : 400 }}>{row.model_name}</span>
                      {row.is_best && <span className="badge" style={{ marginLeft: "10px", backgroundColor: "#3b82f6", color: "#fff", borderColor: "#3b82f6" }}>BEST</span>}
                    </td>
                    <td className="data-mono">{row.score.toFixed(4)}</td>
                    <td className="data-mono">{row.fit_time.toFixed(2)}</td>
                    <td className="data-mono">{row.pred_time.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {(experimentBackend === "unsloth" || (finalMetrics && finalMetrics.retrieval_accuracy !== undefined)) && (
        <section className="card" style={{ marginTop: "2rem", marginBottom: "4rem" }}>
          <h2>{experimentBackend === "unsloth" ? "LLM Chat Interface" : "RAG Chat Interface"}</h2>
          <div className="terminal" style={{ backgroundColor: "#1e1e1e", color: "#d4d4d4", padding: "1.5rem", borderRadius: "8px", minHeight: "300px", display: "flex", flexDirection: "column" }}>
            <div style={{ flex: 1, overflowY: "auto", marginBottom: "1rem" }}>
              {ragChat.map((msg, i) => (
                <div key={i} style={{ marginBottom: "1.5rem" }}>
                  <strong style={{ color: msg.role === "user" ? "#60a5fa" : "#34d399" }}>
                    {msg.role === "user" ? "You: " : "AI: "}
                  </strong>
                  <span style={{ lineHeight: "1.5" }}>{msg.content}</span>
                  {msg.citations && msg.citations.length > 0 && (
                    <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", borderLeft: "2px solid #555", paddingLeft: "10px", color: "#999" }}>
                      <strong>Sources:</strong>
                      <ul style={{ margin: "4px 0 0 0", paddingLeft: "20px" }}>
                        {msg.citations.map((c, idx) => (
                          <li key={idx}>
                            {c.metadata.source} {c.metadata.page ? `(Page ${c.metadata.page})` : ""}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
              {isQuerying && <div style={{ color: "#aaa", fontStyle: "italic" }}>
                {experimentBackend === "unsloth" ? "Loading model (this takes 10-15s on first boot) or generating answer..." : "Generating answer..."}
              </div>}
            </div>
            <form onSubmit={handleChatSubmit} style={{ display: "flex", gap: "10px" }}>
              <input 
                type="text" 
                value={ragQuery}
                onChange={e => setRagQuery(e.target.value)}
                placeholder={experimentBackend === "unsloth" ? "Chat with your fine-tuned model..." : "Ask a question based on your documents..."}
                style={{ flex: 1, padding: "0.8rem", borderRadius: "4px", border: "1px solid #444", background: "#2d2d2d", color: "#fff" }}
                disabled={isQuerying}
              />
              <button type="submit" className="primary" disabled={isQuerying || !ragQuery.trim()}>Ask</button>

            </form>
          </div>
        </section>
      )}
    </main>
  );
}
