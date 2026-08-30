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
    e.currentTarget.classList.add("drag-active");
  };

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove("drag-active");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("drag-active");
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
    <main style={{ padding: "3rem", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ marginBottom: "2rem" }}>
        <h1 style={{ fontSize: "2rem", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "10px" }}>
          <Database size={32} />
          Data Factory (Data Prep)
        </h1>
        <p style={{ color: "#888", fontSize: "1.1rem", maxWidth: "800px" }}>
          Upload a raw unstructured document (PDF, TXT, DOCX). The Data Factory will use <strong>Data-Juicer</strong> to clean and deduplicate the text, and <strong>Distilabel</strong> (powered by <code>nvidia/nemotron-3.5-lightning:free</code> via OpenRouter) to automatically generate conversational Question & Answer pairs. Finally, Microsoft <strong>Presidio</strong> will scrub all Personally Identifiable Information (PII) before returning the final ShareGPT dataset.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
        {/* Left Column: Upload */}
        <section className="card">
          <h2>1. Upload Raw Document</h2>
          
          <div 
            className="dropzone" 
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
            <UploadCloud size={48} style={{ color: "var(--accent)", marginBottom: "1rem" }}/>
            {file ? (
              <p style={{ color: "#fff", fontWeight: 600 }}>{file.name}</p>
            ) : (
              <>
                <p>Drag & Drop a file here, or click to browse</p>
                <small style={{ color: "#666" }}>Supports .txt, .pdf, .docx</small>
              </>
            )}
          </div>
          
          <div style={{ marginTop: "1.5rem" }}>
            <button 
              className="primary" 
              onClick={processFile}
              disabled={!file || isProcessing}
              style={{ width: "100%", justifyContent: "center" }}
            >
              {isProcessing ? "PROCESSING DATA..." : "RUN DATA FACTORY"}
            </button>
          </div>
          
          {error && (
            <div style={{ marginTop: "1rem", color: "#ff4a4a", display: "flex", gap: "8px", alignItems: "center", padding: "1rem", backgroundColor: "rgba(255, 74, 74, 0.1)", borderRadius: "6px" }}>
              <AlertCircle size={20} />
              {error}
            </div>
          )}
        </section>

        {/* Right Column: Results */}
        <section className="card" style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h2>2. Execution Logs & Output</h2>
            {isComplete && (
              <button className="outline" onClick={downloadResult} style={{ padding: "0.5rem 1rem" }}>
                <Download size={16} style={{ marginRight: "8px" }}/>
                Download JSONL
              </button>
            )}
          </div>
          
          <div style={{ 
            backgroundColor: "#000", 
            border: "1px solid #333", 
            borderRadius: "6px", 
            padding: "1rem", 
            overflowY: "auto",
            fontFamily: "'Fira Code', monospace",
            fontSize: "0.85rem",
            color: "#00ffcc",
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
            minHeight: "400px",
            maxHeight: "500px"
          }}>
            {isProcessing || logs.length > 0 ? (
                <>
                {logs.map((log, idx) => (
                    <div key={idx} style={{ marginBottom: "2px", color: log.includes("___FINAL_OUTPUT_PATH___") ? "#00ff00" : "#00ffcc" }}>
                      {log.includes("___FINAL_OUTPUT_PATH___") ? "Pipeline completed successfully!" : log}
                    </div>
                ))}
                <div ref={logsEndRef} />
                </>
            ) : resultData ? (
              <div style={{ color: "#444", display: "flex", height: "100%", justifyContent: "center", alignItems: "center" }}>
                Output ready.
              </div>
            ) : (
              <div style={{ color: "#444", display: "flex", height: "100%", justifyContent: "center", alignItems: "center" }}>
                Waiting to start...
              </div>
            )}
          </div>
          
          {/* Prominent Download Button at the bottom when complete */}
          {isComplete && (
            <div style={{ marginTop: "1.5rem" }}>
              <button 
                className="primary" 
                onClick={downloadResult} 
                style={{ width: "100%", justifyContent: "center", padding: "1rem", fontSize: "1.1rem", backgroundColor: "#00cc88", color: "#000" }}
              >
                <Download size={24} style={{ marginRight: "12px" }}/>
                Download Sanitized Dataset (JSONL)
              </button>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
