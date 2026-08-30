"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Database, FileSpreadsheet, Activity, CheckCircle2, FlaskConical, HelpCircle } from "lucide-react";
import Link from "next/link";

export default function ProjectDashboard() {
  const { id } = useParams();
  const [project, setProject] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showWhy, setShowWhy] = useState(null);

  useEffect(() => {
    const fetchProject = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/projects/${id}`);
        if (res.ok) {
          const data = await res.json();
          setProject(data);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setIsLoading(false);
      }
    };
    fetchProject();
    
    const interval = setInterval(fetchProject, 5000);
    return () => clearInterval(interval);
  }, [id]);

  if (isLoading) return <main><p>Loading project data...</p></main>;
  if (!project) return <main><p>Project not found or backend offline.</p></main>;

  return (
    <main>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1><Database size={24} style={{ display: "inline", marginRight: "12px", color: "var(--color-accent)" }}/>Project: {project.name}</h1>
        <Link href="/" style={{ fontSize: "0.9rem" }}>&larr; Back to Builder</Link>
      </div>
      <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", marginBottom: "3rem" }}>ID: {project.id}</p>

      <section>
        <h2><FileSpreadsheet size={18} style={{ display: "inline", marginRight: "8px" }}/>Raw Data Sources</h2>
        <div className="table-container">
          <table>
            <thead>
              <tr><th>UUID</th><th>Filename</th><th>Type</th><th>Size</th></tr>
            </thead>
            <tbody>
              {project.datasources?.length > 0 ? project.datasources.map(ds => (
                <tr key={ds.id}>
                  <td className="data-mono" style={{ color: "var(--color-text-muted)" }}>{ds.id.substr(0, 8)}...</td>
                  <td className="data-mono">{ds.filename}</td>
                  <td><span className="badge">{ds.type}</span></td>
                  <td className="data-mono">{(ds.size_bytes / 1024).toFixed(1)} KB</td>
                </tr>
              )) : <tr><td colSpan="4">No raw data ingested.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2><Activity size={18} style={{ display: "inline", marginRight: "8px" }}/>Dataset Versions</h2>
        <div className="table-container">
          <table>
            <thead>
              <tr><th>UUID</th><th>Version</th><th>Path</th></tr>
            </thead>
            <tbody>
              {project.datasets?.length > 0 ? project.datasets.map(ds => (
                <tr key={ds.id}>
                  <td className="data-mono" style={{ color: "var(--color-text-muted)" }}>{ds.id.substr(0, 8)}...</td>
                  <td>v{ds.version}</td>
                  <td className="data-mono" style={{ fontSize: "0.75rem" }}>{ds.path}</td>
                </tr>
              )) : <tr><td colSpan="3">No datasets formulated.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2><FlaskConical size={18} style={{ display: "inline", marginRight: "8px" }}/>Experiments</h2>
        <div className="table-container">
          <table>
            <thead>
              <tr><th>UUID</th><th>Model / Backend</th><th>Status</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {project.experiments?.length > 0 ? project.experiments.map(exp => (
                <tr key={exp.id}>
                  <td className="data-mono" style={{ color: "var(--color-text-muted)" }}>{exp.id.substr(0, 8)}...</td>
                  <td className="data-mono">{exp.model_name} <br/><span style={{ color: "var(--color-text-muted)", fontSize: "0.75rem" }}>{exp.backend}</span></td>
                  <td>
                    {exp.status === "completed" && <span style={{ color: "var(--color-accent)", fontWeight: 600 }}><CheckCircle2 size={14} style={{ display: "inline", marginRight: "4px" }}/>COMPLETED</span>}
                    {exp.status !== "completed" && <span style={{ color: "#a1a1aa", fontWeight: 600 }}>{exp.status.toUpperCase()}</span>}
                  </td>
                  <td>
                    <button 
                      onClick={() => setShowWhy(showWhy === exp.id ? null : exp.id)}
                      style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem", background: "transparent" }}
                    >
                      <HelpCircle size={14} style={{ marginRight: "4px" }}/> Why?
                    </button>
                    {showWhy === exp.id && (
                      <div style={{ marginTop: "0.5rem", padding: "0.5rem", background: "var(--color-bg)", border: "1px solid var(--color-border)", borderRadius: "var(--radius)", fontSize: "0.8rem", color: "var(--color-text-muted)", maxWidth: "300px", lineHeight: 1.4 }}>
                        The LLM Orchestrator evaluated your objective and data schema, deciding that `{exp.model_name}` via `{exp.backend}` was the most suitable configuration for this predictive task. 
                        It generated this pipeline securely to meet constraints.
                      </div>
                    )}
                  </td>
                </tr>
              )) : <tr><td colSpan="4">No experiments run.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
