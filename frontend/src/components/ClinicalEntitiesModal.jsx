import React, { useEffect, useState } from "react";
import { getDocumentEntities, getDocumentSanitizedText } from "../api/documents";

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

export default function ClinicalEntitiesModal({ document, onClose }) {
  const [loading, setLoading] = useState(true);
  const [entities, setEntities] = useState([]);
  const [sanitizedText, setSanitizedText] = useState("");
  const [error, setError] = useState(null);

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
        setError(err.message || "Failed to load clinical entities");
        setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [document]);

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
          maxWidth: "800px",
          maxHeight: "85vh",
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
                M4 Clinical Entities
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
                Sanitized Input
              </span>
            </div>
            <p style={{ margin: "4px 0 0 0", fontSize: "0.875rem", color: "#64748b" }}>
              Extracted from {document.file_name || document.document_id} via GLiNER Biomedical NER
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

        {/* Content */}
        <div style={{ padding: "24px", overflowY: "auto", flex: 1 }}>
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
              <p style={{ color: "#64748b", fontSize: "0.875rem" }}>Extracting biomedical entities from sanitized OCR text...</p>
            </div>
          ) : error ? (
            <div style={{ padding: "16px", borderRadius: "8px", background: "#fef2f2", color: "#991b1b" }}>
              {error}
            </div>
          ) : entities.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px 0", color: "#64748b" }}>
              <p>No clinical entities detected in the sanitized text.</p>
            </div>
          ) : (
            <div>
              {/* Stats Bar */}
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

              {/* Grouped Entities */}
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

              {/* Sanitized Text Preview */}
              {sanitizedText && (
                <div style={{ marginTop: "24px" }}>
                  <h4 style={{ margin: "0 0 8px 0", fontSize: "0.875rem", fontWeight: 600, color: "#334155" }}>
                    Sanitized Text Source (De-identified)
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
                      maxHeight: "200px",
                      overflowY: "auto",
                      lineHeight: "1.5",
                    }}
                  >
                    {sanitizedText}
                  </div>
                </div>
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
