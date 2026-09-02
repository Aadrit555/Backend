"use client";

import { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Database,
  FileText,
  Download,
  Copy,
  Check,
  Eye,
  Code,
  ShieldAlert,
  Layers,
  Sparkles,
} from "lucide-react";

export default function DataPrepPage() {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [resultData, setResultData] = useState("");
  const [error, setError] = useState("");

  // Feature 1: Mode selection
  const [mode, setMode] = useState("qa"); // "qa" | "pii_clean" | "rag_chunks"

  // Feature 2: Preview & Stats
  const [previewRecords, setPreviewRecords] = useState([]);
  const [datasetStats, setDatasetStats] = useState(null);
  const [previewTab, setPreviewTab] = useState("cards"); // "cards" | "jsonl"
  const [copied, setCopied] = useState(false);

  const [jobId, setJobId] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isComplete, setIsComplete] = useState(false);

  const fileInputRef = useRef(null);
  const logsEndRef = useRef(null);

  const MODES = [
    {
      id: "qa",
      title: "Q&A PAIRS",
      desc: "Instruction tuning for LLMs",
      icon: Sparkles,
    },
    {
      id: "pii_clean",
      title: "PII SCRUBBING",
      desc: "Redact personal data & keys",
      icon: ShieldAlert,
    },
    {
      id: "rag_chunks",
      title: "RAG CHUNKS",
      desc: "Semantic vector chunks",
      icon: Layers,
    },
  ];

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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const processFile = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError("");
    setResultData("");
    setLogs([]);
    setJobId(null);
    setIsComplete(false);
    setPreviewRecords([]);
    setDatasetStats(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("mode", mode);

    try {
      const res = await fetch("http://localhost:8000/api/data-prep", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Processing failed");
      }

      setJobId(data.job_id);
    } catch (err) {
      console.error(err);
      setError(err.message);
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/data-prep/${jobId}/status`);
        if (!res.ok) return;

        const data = await res.json();
        if (data.logs) {
          setLogs(data.logs);
        }

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
          setIsProcessing(false);
          if (data.status === "completed") {
            setResultData(data.content || "");
            setPreviewRecords(data.preview_records || []);
            setDatasetStats(data.stats || null);
            setIsComplete(true);
          } else {
            setError("Pipeline failed. Check logs.");
          }
        }
      } catch (err) {
        console.error("Status check failed", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [jobId]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const downloadResult = () => {
    if (!resultData) return;
    const blob = new Blob([resultData], { type: "application/jsonl" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${file?.name?.replace(/\.[^/.]+$/, "") || "dataset"}_${mode}.jsonl`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = () => {
    if (!resultData) return;
    navigator.clipboard.writeText(resultData);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <main className="relative z-10 max-w-[1440px] mx-auto px-margin-desktop py-margin-desktop min-h-[calc(100vh-48px)] flex flex-col gap-10">
      {/* Hero Header */}
      <header className="flex flex-col gap-4">
        <span className="text-label-caps font-label-caps text-on-surface-variant">
          DATA FACTORY / 02
        </span>
        <div>
          <h1 className="text-display-lg font-display-lg text-on-background mb-2">
            Prepare your data.
          </h1>
          <p className="text-body-md font-body-md text-on-surface-variant max-w-2xl">
            Clean, deduplicate, and convert raw unstructured documents (PDF, TXT, DOCX) into sanitized datasets for LLM fine-tuning, PII-scrubbed archives, or semantic RAG vector chunking.
          </p>
        </div>
      </header>

      <div className="flex flex-col lg:flex-row gap-8 w-full items-start">
        {/* =============================================================== */}
        {/* Left Column: Configuration & Document Ingestion                 */}
        {/* =============================================================== */}
        <section className="flex-1 flex flex-col gap-5 w-full">
          <h2 className="text-label-caps font-label-caps text-on-surface-variant border-b border-[#1A1A1A] pb-2">
            1. CONFIGURATION & INGESTION
          </h2>

          <div className="bg-[#0A0A0A] border border-[#1A1A1A] p-6 flex flex-col gap-5">
            {/* Feature 1: Mode Selector */}
            <div className="flex flex-col gap-2">
              <label className="text-label-caps font-label-caps text-on-surface-variant">
                PIPELINE PRESET
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {MODES.map((m) => {
                  const isSelected = mode === m.id;
                  const Icon = m.icon;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => setMode(m.id)}
                      className={`p-3 text-left border transition-all flex flex-col gap-1.5 ${
                        isSelected
                          ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white shadow-[0_0_10px_rgba(0,229,255,0.1)]"
                          : "border-[#252525] bg-[#121212] text-[#888888] hover:border-[#444444] hover:text-white"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <Icon size={14} className={isSelected ? "text-[#00E5FF]" : "text-[#777777]"} />
                        <span className="text-label-caps font-label-caps font-bold">
                          {m.title}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-[#666666]">
                        {m.desc}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Document Drag & Drop */}
            <div
              className="border border-dashed border-[#333333] hover:border-[#00E5FF] transition-colors bg-[#131313] p-10 flex flex-col items-center justify-center gap-3 cursor-pointer group min-h-[160px]"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: "none" }}
                onChange={handleFileChange}
                accept=".txt,.pdf,.docx,.html,.csv,.json"
              />
              <UploadCloud
                size={32}
                className="text-[#595959] group-hover:text-[#00E5FF] transition-colors"
              />
              {file ? (
                <div className="text-center flex flex-col gap-1">
                  <span className="text-body-md font-body-md text-[#00E5FF] flex items-center justify-center gap-2 font-bold">
                    <CheckCircle2 size={16} /> {file.name}
                  </span>
                  <span className="text-code-sm font-code-sm text-[#777777]">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              ) : (
                <div className="text-center flex flex-col gap-1">
                  <span className="text-body-md font-body-md text-on-background">
                    Drag & drop your document here
                  </span>
                  <span className="text-code-sm font-code-sm text-[#595959]">
                    or click to browse local files
                  </span>
                  <span className="text-code-sm font-code-sm text-[#595959] mt-1 border border-[#333333] px-2 py-0.5 bg-[#1A1A1A] self-center">
                    PDF · TXT · DOCX · HTML
                  </span>
                </div>
              )}
            </div>

            <button
              className="bg-[#00E5FF] text-black w-full py-3 text-label-caps font-label-caps font-bold hover:bg-white transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              onClick={processFile}
              disabled={!file || isProcessing}
            >
              {isProcessing ? "PROCESSING PIPELINE..." : "RUN DATA FACTORY"}
              {!isProcessing && (
                <span className="material-symbols-outlined text-sm" data-icon="arrow_forward">
                  arrow_forward
                </span>
              )}
            </button>

            {error && (
              <div className="p-3 border border-[#93000a] bg-[#1a0002] text-[#ffb4ab] text-code-sm font-code-sm flex items-start gap-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {/* =============================================================== */}
        {/* Right Column: Execution Logs & Feature 2 Live Dataset Inspector  */}
        {/* =============================================================== */}
        <section className="flex-1 flex flex-col gap-5 w-full">
          <div className="flex justify-between items-end border-b border-[#1A1A1A] pb-2 flex-wrap gap-2">
            <h2 className="text-label-caps font-label-caps text-on-surface-variant">
              2. EXECUTION & DATASET PROFILER
            </h2>

            {isComplete && (
              <div className="flex items-center gap-3">
                <button
                  onClick={copyToClipboard}
                  className="text-label-caps font-label-caps text-on-surface-variant hover:text-white transition-colors flex items-center gap-1 text-xs"
                >
                  {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                  {copied ? "[ COPIED ]" : "[ COPY ALL ]"}
                </button>
                <button
                  onClick={downloadResult}
                  className="text-label-caps font-label-caps text-[#00E5FF] hover:text-white transition-colors flex items-center gap-1 text-xs"
                >
                  <Download size={12} />
                  [ DOWNLOAD JSONL ]
                </button>
              </div>
            )}
          </div>

          <div className="bg-[#0A0A0A] border border-[#1A1A1A] p-4 flex flex-col min-h-[460px] gap-3">
            {/* Feature 2: Stats & Inspector Tabs when complete */}
            {isComplete && datasetStats && (
              <div className="flex flex-col gap-3 pb-3 border-b border-[#1C1C1C]">
                {/* Stats Bar */}
                <div className="flex items-center gap-4 text-xs font-mono bg-[#121212] p-2.5 border border-[#222222] flex-wrap">
                  <div className="flex items-center gap-1.5 text-white">
                    <CheckCircle2 size={13} className="text-emerald-400" />
                    <span>STATUS: <strong>READY</strong></span>
                  </div>
                  <div className="text-[#888888]">
                    RECORDS: <strong className="text-white">{datasetStats.total_records}</strong>
                  </div>
                  <div className="text-[#888888]">
                    EST. TOKENS: <strong className="text-[#00E5FF]">~{datasetStats.estimated_tokens?.toLocaleString()}</strong>
                  </div>
                  <div className="text-[#888888]">
                    FILE SIZE: <strong className="text-white">{datasetStats.file_size_kb} KB</strong>
                  </div>
                </div>

                {/* View switcher & Action buttons */}
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setPreviewTab("cards")}
                      className={`px-2.5 py-1 text-label-caps font-label-caps transition-colors flex items-center gap-1.5 border cursor-pointer ${
                        previewTab === "cards"
                          ? "border-[#00E5FF] bg-[#00E5FF]/15 text-[#00E5FF] font-bold"
                          : "border-[#252525] bg-[#141414] text-[#777777] hover:text-white"
                      }`}
                    >
                      <Eye size={12} /> CARDS VIEW
                    </button>
                    <button
                      type="button"
                      onClick={() => setPreviewTab("jsonl")}
                      className={`px-2.5 py-1 text-label-caps font-label-caps transition-colors flex items-center gap-1.5 border cursor-pointer ${
                        previewTab === "jsonl"
                          ? "border-[#00E5FF] bg-[#00E5FF]/15 text-[#00E5FF] font-bold"
                          : "border-[#252525] bg-[#141414] text-[#777777] hover:text-white"
                      }`}
                    >
                      <Code size={12} /> RAW JSONL
                    </button>
                  </div>

                  {/* Prominent Action Buttons */}
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={copyToClipboard}
                      className="px-3 py-1 text-label-caps font-label-caps transition-all flex items-center gap-1.5 border border-[#333333] bg-[#161616] text-[#CCCCCC] hover:text-white hover:border-[#666666] text-xs cursor-pointer"
                    >
                      {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                      <span>{copied ? "[ COPIED ]" : "[ COPY ALL ]"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={downloadResult}
                      className="px-3.5 py-1.5 text-label-caps font-label-caps font-bold transition-all flex items-center gap-1.5 bg-[#00E5FF] text-black hover:bg-white text-xs cursor-pointer shadow-[0_0_12px_rgba(0,229,255,0.3)]"
                    >
                      <Download size={14} className="text-black stroke-[2.5]" />
                      <span>[ DOWNLOAD JSONL ]</span>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Main Area: Preview vs Logs */}
            <div className="flex-1 overflow-y-auto max-h-[380px]">
              {isProcessing || (!isComplete && logs.length > 0) ? (
                <div className="flex flex-col gap-1 font-code-sm text-code-sm">
                  {logs.map((log, idx) => {
                    const isSuccess = log.includes("___FINAL_OUTPUT_PATH___");
                    return (
                      <div
                        key={idx}
                        className={isSuccess ? "text-[#00E5FF]" : "text-on-surface-variant"}
                      >
                        {isSuccess ? "[COMPLETED] Pipeline finished successfully." : `> ${log}`}
                      </div>
                    );
                  })}
                  <div ref={logsEndRef} />
                </div>
              ) : isComplete && previewRecords.length > 0 ? (
                previewTab === "cards" ? (
                  <div className="flex flex-col gap-2.5 pr-1">
                    {previewRecords.map((item, idx) => (
                      <div
                        key={idx}
                        className="bg-[#121212] border border-[#222222] p-3 text-xs font-mono flex flex-col gap-2"
                      >
                        <div className="flex justify-between text-[#555555] text-[10px] border-b border-[#1E1E1E] pb-1">
                          <span>RECORD #{idx + 1}</span>
                          <span>{item.source || "Cleaned Document"}</span>
                        </div>

                        {/* Conversational Q&A format */}
                        {item.conversations ? (
                          <div className="flex flex-col gap-1.5">
                            {item.conversations.map((c, cIdx) => (
                              <div key={cIdx} className="flex flex-col gap-0.5">
                                <span
                                  className={`text-[10px] font-bold ${
                                    c.from === "human" ? "text-[#00E5FF]" : "text-emerald-400"
                                  }`}
                                >
                                  {c.from === "human" ? "QUESTION:" : "RESPONSE:"}
                                </span>
                                <p className="text-white whitespace-pre-wrap pl-1 border-l border-[#222222]">
                                  {c.value}
                                </p>
                              </div>
                            ))}
                          </div>
                        ) : (
                          /* Text chunk or sanitized text */
                          <div className="text-white whitespace-pre-wrap">
                            {item.text || JSON.stringify(item)}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <pre className="text-code-sm font-code-sm text-on-surface-variant p-2 bg-black border border-[#1C1C1C] overflow-x-auto whitespace-pre-wrap">
                    {resultData}
                  </pre>
                )
              ) : (
                <div className="h-full flex items-center justify-center text-[#595959] text-code-sm font-code-sm italic py-24">
                  Select a preset, upload a document, and click [ RUN DATA FACTORY ].
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
