import React, { useEffect, useState } from "react";
import {
  getDocumentEntities,
  getDocumentSanitizedText,
  classifyDocument,
  summarizeDocument,
} from "../api/documents";

const LABEL_COLORS = {
  disease: { bg: "#fee2e2", text: "#991b1b", border: "#fca5a5" },
  medication: { bg: "#dbeafe", text: "#1e40af", border: "#93c5fd" },
  dosage: { bg: "#e0e7ff", text: "#3730a3", border: "#a5b4fc" },
  symptom: { bg: "#fef3c7", text: "#92400e", border: "#fde68a" },
  procedure: { bg: "#f3e8ff", text: "#6b21a8", border: "#d8b4fe" },
  "anatomical structure": { bg: "#dcfce7", text: "#166534", border: "#86efac" },
  "lab test": { bg: "#ccfbf1", text: "#115e59", border: "#5eead4" },
  "lab value": { bg: "#fef9c3", text: "#854d0e", border: "#fde047" },
};

export default function ClinicalEntitiesModal({ document, onClose, onDocumentUpdated }) {
  const [loading, setLoading] = useState(true);
  const [entities, setEntities] = useState([]);
  const [sanitizedText, setSanitizedText] = useState("");
  const [error, setError] = useState(null);

  // M3 and M5 states
  const [docType, setDocType] = useState(document?.document_type || null);
  const [docConfidence, setDocConfidence] = useState(document?.classification_confidence ?? null);
  const [summary, setSummary] = useState(document?.summary || null);
  const [keyFindings, setKeyFindings] = useState(document?.key_findings || []);
  const [isClassifying, setIsClassifying] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    if (!document?.document_id) return;
    let isMounted = true;
    setLoading(true);
    setError(null);

    Promise.all([
      getDocumentEntities(document.document_id).catch(() => ({ entities: [], total_count: 0 })),
      getDocumentSanitizedText(document.document_id).catch(() => ({ sanitized_text: "" })),
    ])
      .then(([entitiesRes, textRes]) => {
        if (!isMounted) return;
        setEntities(entitiesRes.entities || []);
        setSanitizedText(textRes.sanitized_text || "");
        setLoading(false);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err.message || "Failed to load clinical information");
        setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [document]);

  const handleClassify = async () => {
    setIsClassifying(true);
    setError(null);
    try {
      const res = await classifyDocument(document.document_id);
      setDocType(res.document_type);
      setDocConfidence(res.classification_confidence);
      if (onDocumentUpdated) {
        onDocumentUpdated({
          ...document,
          document_type: res.document_type,
          classification_confidence: res.classification_confidence,
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to classify document");
    } finally {
      setIsClassifying(false);
    }
  };

  const handleSummarize = async () => {
    setIsSummarizing(true);
    setError(null);
    try {
      const res = await summarizeDocument(document.document_id);
      setSummary(res.summary);
      setKeyFindings(res.key_findings || []);
      if (onDocumentUpdated) {
        onDocumentUpdated({
          ...document,
          summary: res.summary,
          key_findings: res.key_findings,
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to summarize document");
    } finally {
      setIsSummarizing(false);
    }
  };

  if (!document) return null;

  // Group entities by label
  const grouped = entities.reduce((acc, ent) => {
    const label = ent.label || "other";
    if (!acc[label]) acc[label] = [];
    acc[label].push(ent);
    return acc;
  }, {});

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(15, 23, 42, 0.65)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: "20px",
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "#ffffff",
          borderRadius: "16px",
          width: "100%",
          maxWidth: "850px",
          maxHeight: "90vh",
          display: "flex",
          flexDirection: "column",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
          overflow: "hidden",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: "20px 24px",
            borderBottom: "1px solid var(--hp-border, #e2e8f0)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "var(--hp-surface-subtle, #f8fafc)",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <h2 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 600, color: "#0f172a" }}>
                Clinical Document Intelligence
              </h2>
              <span
                style={{
                  fontSize: "0.75rem",
                  padding: "2px 8px",
                  borderRadius: "12px",
                  background: "#dcfce7",
                  color: "#166534",
                  fontWeight: 600,
                }}
              >
                Privacy Protected (M3 / M4 / M5)
              </span>
            </div>
            <p style={{ margin: "4px 0 0 0", fontSize: "0.875rem", color: "#64748b" }}>
              {document.file_name || document.document_id}
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "8px",
              borderRadius: "8px",
              color: "#64748b",
            }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        {/* Tab Navigation */}
        <div
          style={{
            display: "flex",
            borderBottom: "1px solid #e2e8f0",
            background: "#ffffff",
            padding: "0 24px",
          }}
        >
          <button
            onClick={() => setActiveTab("overview")}
            style={{
              padding: "12px 16px",
              border: "none",
              background: "none",
              borderBottom: activeTab === "overview" ? "2px solid var(--hp-primary, #0284c7)" : "2px solid transparent",
              color: activeTab === "overview" ? "var(--hp-primary, #0284c7)" : "#64748b",
              fontWeight: 600,
              fontSize: "0.875rem",
              cursor: "pointer",
            }}
          >
            AI Summary & Type (M3/M5)
          </button>
          <button
            onClick={() => setActiveTab("entities")}
            style={{
              padding: "12px 16px",
              border: "none",
              background: "none",
              borderBottom: activeTab === "entities" ? "2px solid var(--hp-primary, #0284c7)" : "2px solid transparent",
              color: activeTab === "entities" ? "var(--hp-primary, #0284c7)" : "#64748b",
              fontWeight: 600,
              fontSize: "0.875rem",
              cursor: "pointer",
            }}
          >
            Clinical Entities (M4) ({entities.length})
          </button>
          <button
            onClick={() => setActiveTab("sanitized")}
            style={{
              padding: "12px 16px",
              border: "none",
              background: "none",
              borderBottom: activeTab === "sanitized" ? "2px solid var(--hp-primary, #0284c7)" : "2px solid transparent",
              color: activeTab === "sanitized" ? "var(--hp-primary, #0284c7)" : "#64748b",
              fontWeight: 600,
              fontSize: "0.875rem",
              cursor: "pointer",
            }}
          >
            Sanitized Source
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: "24px", overflowY: "auto", flex: 1 }}>
          {error && (
            <div
              style={{
                padding: "12px 16px",
                marginBottom: "16px",
                borderRadius: "8px",
                background: "#fef2f2",
                color: "#991b1b",
                fontSize: "0.875rem",
              }}
            >
              {error}
            </div>
          )}

          {loading ? (
            <div style={{ textAlign: "center", padding: "40px 0" }}>
              <div
                className="hp-spinner"
                style={{
                  width: 32,
                  height: 32,
                  margin: "0 auto 16px auto",
                  borderTopColor: "var(--hp-primary, #0284c7)",
                  borderColor: "#e2e8f0",
                }}
              />
              <p style={{ color: "#64748b", fontSize: "0.875rem" }}>Loading document analysis...</p>
            </div>
          ) : activeTab === "overview" ? (
            <div>
              {/* M3 Section: Document Type */}
              <div
                style={{
                  background: "#f8fafc",
                  borderRadius: "12px",
                  padding: "16px 20px",
                  marginBottom: "20px",
                  border: "1px solid #e2e8f0",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <span style={{ fontSize: "0.75rem", textTransform: "uppercase", fontWeight: 700, color: "#64748b" }}>
                    M3 Document Type Identification
                  </span>
                  <div style={{ marginTop: "4px", display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ fontSize: "1.125rem", fontWeight: 700, color: "#0f172a" }}>
                      {docType || "Unclassified"}
                    </span>
                    {docConfidence != null && (
                      <span
                        style={{
                          fontSize: "0.75rem",
                          padding: "2px 8px",
                          borderRadius: "12px",
                          background: "#e0f2fe",
                          color: "#0369a1",
                          fontWeight: 600,
                        }}
                      >
                        {(docConfidence * 100).toFixed(1)}% confidence
                      </span>
                    )}
                  </div>
                </div>

                {!docType && (
                  <button
                    onClick={handleClassify}
                    disabled={isClassifying}
                    style={{
                      background: "var(--hp-primary, #0284c7)",
                      color: "#ffffff",
                      border: "none",
                      padding: "8px 16px",
                      borderRadius: "8px",
                      fontSize: "0.8125rem",
                      fontWeight: 600,
                      cursor: isClassifying ? "not-allowed" : "pointer",
                      opacity: isClassifying ? 0.7 : 1,
                    }}
                  >
                    {isClassifying ? "Classifying..." : "Identify Type"}
                  </button>
                )}
              </div>

              {/* M5 Section: Summarization & Key Findings */}
              <div
                style={{
                  background: "#f8fafc",
                  borderRadius: "12px",
                  padding: "20px",
                  border: "1px solid #e2e8f0",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                  <span style={{ fontSize: "0.75rem", textTransform: "uppercase", fontWeight: 700, color: "#64748b" }}>
                    M5 Healthcare Document Summary
                  </span>
                  {!summary && (
                    <button
                      onClick={handleSummarize}
                      disabled={isSummarizing}
                      style={{
                        background: "var(--hp-primary, #0284c7)",
                        color: "#ffffff",
                        border: "none",
                        padding: "8px 16px",
                        borderRadius: "8px",
                        fontSize: "0.8125rem",
                        fontWeight: 600,
                        cursor: isSummarizing ? "not-allowed" : "pointer",
                        opacity: isSummarizing ? 0.7 : 1,
                      }}
                    >
                      {isSummarizing ? "Generating Summary..." : "Generate AI Summary"}
                    </button>
                  )}
                </div>

                {summary ? (
                  <div>
                    <p style={{ margin: "0 0 16px 0", fontSize: "0.9375rem", lineHeight: "1.6", color: "#1e293b" }}>
                      {summary}
                    </p>

                    {keyFindings && keyFindings.length > 0 && (
                      <div>
                        <h4 style={{ margin: "0 0 10px 0", fontSize: "0.8125rem", textTransform: "uppercase", color: "#475569", fontWeight: 700 }}>
                          Key Findings & Measurements
                        </h4>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                          {keyFindings.map((finding, idx) => (
                            <span
                              key={idx}
                              style={{
                                display: "inline-block",
                                padding: "6px 12px",
                                borderRadius: "8px",
                                background: "#ffffff",
                                border: "1px solid #cbd5e1",
                                fontSize: "0.8125rem",
                                color: "#0f172a",
                                fontWeight: 500,
                              }}
                            >
                              {finding}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p style={{ margin: 0, fontSize: "0.875rem", color: "#64748b" }}>
                    No AI summary generated yet. Click "Generate AI Summary" to create a grounded clinical summary.
                  </p>
                )}
              </div>
            </div>
          ) : activeTab === "entities" ? (
            <div>
              {entities.length === 0 ? (
                <div style={{ textAlign: "center", padding: "40px 0", color: "#64748b" }}>
                  <p>No clinical entities detected in the sanitized text.</p>
                </div>
              ) : (
                <div>
                  <div
                    style={{
                      display: "flex",
                      gap: "16px",
                      marginBottom: "20px",
                      padding: "12px 16px",
                      borderRadius: "8px",
                      background: "#f1f5f9",
                      fontSize: "0.875rem",
                      color: "#334155",
                      fontWeight: 500,
                    }}
                  >
                    <span>Total Entities Found: <strong>{entities.length}</strong></span>
                    <span>Categories: <strong>{Object.keys(grouped).length}</strong></span>
                  </div>

                  {Object.entries(grouped).map(([label, items]) => {
                    const style = LABEL_COLORS[label.toLowerCase()] || { bg: "#f1f5f9", text: "#334155", border: "#cbd5e1" };
                    return (
                      <div key={label} style={{ marginBottom: "20px" }}>
                        <div
                          style={{
                            textTransform: "uppercase",
                            fontSize: "0.75rem",
                            fontWeight: 700,
                            letterSpacing: "0.05em",
                            color: style.text,
                            marginBottom: "8px",
                            display: "flex",
                            alignItems: "center",
                            gap: "6px",
                          }}
                        >
                          <span
                            style={{
                              width: "8px",
                              height: "8px",
                              borderRadius: "50%",
                              background: style.text,
                            }}
                          />
                          {label} ({items.length})
                        </div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                          {items.map((ent, idx) => (
                            <div
                              key={idx}
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: "8px",
                                padding: "6px 12px",
                                borderRadius: "8px",
                                backgroundColor: style.bg,
                                border: `1px solid ${style.border}`,
                                color: style.text,
                                fontSize: "0.875rem",
                                fontWeight: 500,
                              }}
                            >
                              <span>{ent.text}</span>
                              <span
                                style={{
                                  fontSize: "0.7rem",
                                  opacity: 0.85,
                                  fontFeatureSettings: '"tnum"',
                                  fontVariantNumeric: "tabular-nums",
                                  background: "rgba(255,255,255,0.6)",
                                  padding: "2px 6px",
                                  borderRadius: "4px",
                                }}
                              >
                                {(ent.confidence * 100).toFixed(1)}% conf
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ) : (
            <div>
              {sanitizedText ? (
                <div>
                  <h4 style={{ margin: "0 0 8px 0", fontSize: "0.875rem", fontWeight: 600, color: "#334155" }}>
                    Sanitized Text Source (Strict LLM Input Boundary)
                  </h4>
                  <div
                    style={{
                      background: "#0f172a",
                      color: "#f8fafc",
                      padding: "16px",
                      borderRadius: "8px",
                      fontSize: "0.8125rem",
                      fontFamily: "monospace",
                      whiteSpace: "pre-wrap",
                      maxHeight: "350px",
                      overflowY: "auto",
                      lineHeight: "1.5",
                    }}
                  >
                    {sanitizedText}
                  </div>
                </div>
              ) : (
                <p style={{ color: "#64748b" }}>Sanitized text is not available.</p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: "16px 24px",
            borderTop: "1px solid var(--hp-border, #e2e8f0)",
            display: "flex",
            justifyContent: "flex-end",
            background: "var(--hp-surface-subtle, #f8fafc)",
          }}
        >
          <button
            onClick={onClose}
            style={{
              padding: "8px 20px",
              borderRadius: "8px",
              background: "#0f172a",
              color: "#ffffff",
              border: "none",
              fontWeight: 500,
              cursor: "pointer",
              fontSize: "0.875rem",
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
