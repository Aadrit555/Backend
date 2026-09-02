"use client";

import { useState, useRef, useEffect } from "react";
import { UploadCloud, CheckCircle2, AlertCircle, Database, FileText, Download, ArrowRight, Terminal } from "lucide-react";

export default function DataPrepPage() {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [resultData, setResultData] = useState("");
  const [error, setError] = useState("");

  const [jobId, setJobId] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isComplete, setIsComplete] = useState(false);

  const fileInputRef = useRef(null);
  const logsEndRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
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

    const formData = new FormData();
    formData.append("file", file);

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
    a.download = "sanitized_dataset.jsonl";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <main className="relative z-10 max-w-[1360px] mx-auto px-5 py-5 flex flex-col gap-5 text-[#E0E0E0]">
      {/* Top Header */}
      <div className="flex items-center justify-between border-b border-[#222222] pb-3 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Database size={16} className="text-[#00E5FF]" />
          <h1 className="text-sm font-mono font-bold text-white uppercase tracking-wider">
            DATA FACTORY
          </h1>
          <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1C1C1C] text-[#888888] border border-[#2A2A2A]">
            CLEANING & PII SCRUBBING
          </span>
        </div>
        <span className="text-xs font-mono text-[#666666]">
          Automated text sanitization, PII redaction, and Q&A formatting
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column: Ingestion */}
        <section className="lg:col-span-6 flex flex-col gap-3">
          <div className="bg-[#0F0F0F] border border-[#222222] p-4 flex flex-col gap-4">
            <span className="text-xs font-mono text-[#888888] uppercase">
              1. Document Ingestion
            </span>

            <div
              className="border border-dashed border-[#333333] hover:border-[#00E5FF] transition-colors bg-[#141414] p-10 flex flex-col items-center justify-center gap-3 cursor-pointer group"
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
                accept=".txt,.pdf,.docx,.html"
              />
              <UploadCloud
                size={30}
                className="text-[#666666] group-hover:text-[#00E5FF] transition-colors"
              />

              {file ? (
                <div className="text-center flex flex-col gap-0.5">
                  <span className="text-xs font-mono text-[#00E5FF] font-bold flex items-center justify-center gap-1.5">
                    <CheckCircle2 size={13} /> {file.name}
                  </span>
                  <span className="text-[11px] font-mono text-[#666666]">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              ) : (
                <div className="text-center flex flex-col gap-1">
                  <span className="text-xs font-mono text-white">
                    Drop document here or click to browse
                  </span>
                  <span className="text-[10px] font-mono text-[#666666]">
                    PDF · TXT · DOCX · HTML
                  </span>
                </div>
              )}
            </div>

            <button
              className="bg-[#00E5FF] hover:bg-[#00cbe2] text-black w-full py-2.5 px-4 text-xs font-mono uppercase font-bold transition-all flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer shadow-[0_0_15px_rgba(0,229,255,0.15)]"
              onClick={processFile}
              disabled={!file || isProcessing}
            >
              {isProcessing ? "PROCESSING PIPELINE..." : "RUN DATA FACTORY"}
              <ArrowRight size={13} />
            </button>

            {error && (
              <div className="p-2.5 border border-red-800 bg-red-950/40 text-red-300 text-xs font-mono flex items-start gap-2">
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {/* Right Column: Execution Logs & Output */}
        <section className="lg:col-span-6 flex flex-col gap-3">
          <div className="bg-[#0F0F0F] border border-[#222222] p-4 flex flex-col gap-3 min-h-[360px]">
            <div className="flex justify-between items-center border-b border-[#1E1E1E] pb-2">
              <span className="text-xs font-mono text-[#888888] flex items-center gap-1.5 uppercase">
                <Terminal size={12} className="text-[#00E5FF]" /> 2. Pipeline Logs
              </span>
              {isComplete && (
                <button
                  onClick={downloadResult}
                  className="text-xs font-mono text-[#00E5FF] hover:underline flex items-center gap-1"
                >
                  <Download size={12} /> Download JSONL
                </button>
              )}
            </div>

            <div className="flex-1 bg-black border border-[#1C1C1C] p-3 overflow-y-auto font-mono text-xs max-h-[320px]">
              {isProcessing || logs.length > 0 ? (
                <div className="flex flex-col gap-1">
                  {logs.map((log, idx) => {
                    const isSuccess = log.includes("___FINAL_OUTPUT_PATH___");
                    return (
                      <div
                        key={idx}
                        className={isSuccess ? "text-[#00E5FF] font-bold" : "text-[#888888]"}
                      >
                        {isSuccess ? "✓ Pipeline finished successfully." : `> ${log}`}
                      </div>
                    );
                  })}
                  <div ref={logsEndRef} />
                </div>
              ) : resultData ? (
                <div className="h-full flex items-center justify-center text-[#00E5FF]">
                  ✓ Output ready for download.
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-[#555555]">
                  Logs will appear here once processing begins.
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
