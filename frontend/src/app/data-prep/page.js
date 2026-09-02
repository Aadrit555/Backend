"use client";

import { useState, useRef, useEffect } from "react";
import { UploadCloud, CheckCircle2, AlertCircle, Database, FileText, Download } from "lucide-react";

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
    const blob = new Blob([resultData], { type: 'application/jsonl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sanitized_dataset.jsonl';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <main className="relative z-10 max-w-[1440px] mx-auto px-margin-desktop py-margin-desktop min-h-[calc(100vh-48px)] flex flex-col gap-12">
      
      {/* Hero Header */}
      <header className="flex flex-col gap-4">
        <span className="text-label-caps font-label-caps text-on-surface-variant">DATA FACTORY / 02</span>
        <div>
          <h1 className="text-display-lg font-display-lg text-on-background mb-2">Prepare your data.</h1>
          <p className="text-body-md font-body-md text-on-surface-variant max-w-xl">
            Upload raw unstructured documents (PDF, TXT, DOCX). We use <strong className="text-[#00E5FF]">Data-Juicer</strong> to clean text, <strong className="text-[#00E5FF]">Distilabel</strong> for Q&A generation, and <strong className="text-[#00E5FF]">Presidio</strong> to automatically scrub PII.
          </p>
        </div>
      </header>

      <div className="flex flex-col lg:flex-row gap-12 w-full">
        {/* Left Column: Upload */}
        <section className="flex-1 flex flex-col gap-6">
          <h2 className="text-label-caps font-label-caps text-on-surface-variant border-b border-[#1A1A1A] pb-2">1. RAW DOCUMENT INGESTION</h2>
          
          <div className="bg-[#0A0A0A] border border-[#1A1A1A] p-6 flex flex-col gap-4 h-full">
            <div 
              className="border border-dashed border-[#333333] hover:border-[#00E5FF] transition-colors bg-[#131313] p-12 flex flex-col items-center justify-center gap-4 cursor-pointer group flex-grow"
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
              <span className="material-symbols-outlined text-4xl text-[#595959] group-hover:text-[#00E5FF] transition-colors" data-icon="cloud_upload">cloud_upload</span>
              {file ? (
                <div className="text-center flex flex-col gap-1">
                  <span className="text-body-md font-body-md text-[#00E5FF] flex items-center gap-2">
                    <CheckCircle2 size={16} /> {file.name}
                  </span>
                  <span className="text-code-sm font-code-sm text-[#595959]">{(file.size / 1024).toFixed(1)} KB</span>
                </div>
              ) : (
                <div className="text-center flex flex-col gap-1">
                  <span className="text-body-md font-body-md text-on-background">
                    Drag & Drop your document here
                  </span>
                  <span className="text-code-sm font-code-sm text-[#595959]">or click to browse local files</span>
                  <span className="text-code-sm font-code-sm text-[#595959] mt-2 border border-[#333333] px-2 py-1 bg-[#1A1A1A] self-center">
                    PDF · TXT · DOCX
                  </span>
                </div>
              )}
            </div>
            
            <button 
              className="bg-[#00E5FF] text-black w-full py-3 text-label-caps font-label-caps font-bold hover:bg-white transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={processFile}
              disabled={!file || isProcessing}
            >
              {isProcessing ? "PROCESSING PIPELINE..." : "RUN DATA FACTORY"} 
              {!isProcessing && <span className="material-symbols-outlined text-sm" data-icon="arrow_forward">arrow_forward</span>}
            </button>
            
            {error && (
              <div className="mt-4 p-4 border border-[#93000a] bg-[#1a0002] text-[#ffb4ab] text-code-sm font-code-sm flex items-start gap-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {/* Right Column: Results */}
        <section className="flex-1 flex flex-col gap-6">
          <div className="flex justify-between items-end border-b border-[#1A1A1A] pb-2">
            <h2 className="text-label-caps font-label-caps text-on-surface-variant">2. EXECUTION LOGS & OUTPUT</h2>
            {isComplete && (
              <button 
                onClick={downloadResult} 
                className="text-label-caps font-label-caps text-[#00E5FF] hover:text-white transition-colors flex items-center gap-1"
              >
                [ DOWNLOAD JSONL ]
              </button>
            )}
          </div>
          
          <div className="bg-[#0e0e0e] border border-[#1A1A1A] p-4 flex flex-col h-full min-h-[400px]">
            <div className="flex-1 overflow-y-auto font-code-sm text-code-sm">
              {isProcessing || logs.length > 0 ? (
                  <div className="flex flex-col gap-1">
                  {logs.map((log, idx) => {
                      const isSuccess = log.includes("___FINAL_OUTPUT_PATH___");
                      return (
                        <div key={idx} className={`${isSuccess ? "text-[#00E5FF]" : "text-on-surface-variant"}`}>
                          {isSuccess ? "[COMPLETED] Pipeline finished successfully." : `> ${log}`}
                        </div>
                      );
                  })}
                  <div ref={logsEndRef} />
                  </div>
              ) : resultData ? (
                <div className="h-full flex items-center justify-center text-[#00E5FF]">
                  [ OUTPUT READY FOR DOWNLOAD ]
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-[#595959] italic">
                  Waiting for pipeline initialization...
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
