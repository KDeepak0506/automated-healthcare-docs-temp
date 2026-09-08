/**
 * PatientWorkspacePage
 *
 * A split-panel patient workspace:
 *   Left panel  — Patient demographics, documents list, upload action
 *   Right panel — Clinical AI Assistant scoped to the patient
 *
 * Route: /patients/:patientId
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams, Link } from "react-router-dom";
import { getPatient } from "../api/patients";
import { listDocuments, uploadDocument, getDocumentStatus } from "../api/documents";
import { searchPatientAI } from "../api/patients";
import Toast from "../components/Toast";

/* ─────────────────── helpers ─────────────────── */
const POLL_INTERVAL_MS = 4000;
const ACTIVE_STATUSES = ["Pending", "Processing"];
const STATUS_COLORS = {
  Completed: "var(--hp-success)",
  Processing: "var(--hp-warning)",
  Pending: "var(--hp-warning)",
  Failed: "var(--hp-danger)",
};

function fmtDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function calcAge(dob) {
  if (!dob) return null;
  const birth = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  if (
    today.getMonth() < birth.getMonth() ||
    (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate())
  )
    age--;
  return age;
}

/* ─────────────────── StatusBadge ─────────────────── */
function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || "var(--hp-text-500)";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        fontSize: "0.75rem",
        fontWeight: 600,
        color,
        background: color + "18",
        borderRadius: 20,
        padding: "2px 10px",
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: color,
          animation:
            status === "Processing" || status === "Pending"
              ? "hp-pulse 1.5s ease-in-out infinite"
              : "none",
        }}
      />
      {status}
    </span>
  );
}

/* ─────────────────── AIMessage ─────────────────── */
function AIMessage({ role, text, sources }) {
  return (
    <div
      className={`pw-ai-message pw-ai-${role}`}
      style={{
        display: "flex",
        gap: 10,
        alignItems: "flex-start",
        animation: "hp-fade-in 0.25s ease",
      }}
    >
      <div
        style={{
          flexShrink: 0,
          width: 30,
          height: 30,
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "0.75rem",
          fontWeight: 700,
          background:
            role === "assistant"
              ? "linear-gradient(135deg, var(--hp-primary-600), var(--hp-primary-400))"
              : "var(--hp-bg-300)",
          color: role === "assistant" ? "#fff" : "var(--hp-text-300)",
        }}
      >
        {role === "assistant" ? "AI" : "You"}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p
          style={{
            background:
              role === "assistant" ? "var(--hp-bg-200)" : "var(--hp-bg-300)",
            borderRadius: role === "assistant" ? "4px 12px 12px 12px" : "12px 4px 12px 12px",
            padding: "10px 14px",
            margin: 0,
            fontSize: "0.875rem",
            lineHeight: 1.6,
            color: "var(--hp-text-100)",
            whiteSpace: "pre-wrap",
          }}
        >
          {text}
        </p>
        {sources && sources.length > 0 && (
          <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 4 }}>
            {sources.map((s, i) => (
              <span
                key={i}
                title={s.text}
                style={{
                  fontSize: "0.7rem",
                  padding: "2px 8px",
                  background: "var(--hp-primary-600)22",
                  border: "1px solid var(--hp-primary-600)44",
                  borderRadius: 10,
                  color: "var(--hp-primary-400)",
                  cursor: "default",
                }}
              >
                Source {i + 1}{s.page_number ? ` · p.${s.page_number}` : ""}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ─────────────────── Main Component ─────────────────── */
export default function PatientWorkspacePage() {
  const { patientId } = useParams();
  const navigate = useNavigate();

  const [patient, setPatient] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loadingPatient, setLoadingPatient] = useState(true);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState(null);

  // AI state
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hello! I'm your Clinical AI Assistant for this patient. You can ask me about their documents, lab results, medication history, or any clinical questions.",
      sources: [],
    },
  ]);
  const [aiQuery, setAiQuery] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const chatBottomRef = useRef(null);
  const pollRef = useRef(null);
  const fileInputRef = useRef(null);

  /* Load patient */
  useEffect(() => {
    (async () => {
      try {
        const data = await getPatient(patientId);
        setPatient(data);
      } catch {
        setToast({ message: "Patient not found or access denied.", variant: "error" });
        setTimeout(() => navigate("/patients"), 2000);
      } finally {
        setLoadingPatient(false);
      }
    })();
  }, [patientId, navigate]);

  /* Load documents */
  const fetchDocs = useCallback(async () => {
    try {
      const data = await listDocuments({ patient_id: patientId });
      const items = Array.isArray(data) ? data : data?.items ?? [];
      setDocuments(items);
    } catch {
      // silent
    } finally {
      setLoadingDocs(false);
    }
  }, [patientId]);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  /* Poll active docs */
  useEffect(() => {
    const activeDocs = documents.filter(
      (d) =>
        ACTIVE_STATUSES.includes(d.processing_status) ||
        ["pending", "processing"].includes(d.privacy_status)
    );
    if (activeDocs.length === 0) {
      clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(async () => {
      const updates = await Promise.all(
        activeDocs.map((d) => getDocumentStatus(d.document_id).catch(() => null))
      );
      setDocuments((prev) =>
        prev.map((doc) => {
          const u = updates.find((x) => x && x.document_id === doc.document_id);
          return u ? { ...doc, processing_status: u.status, privacy_status: u.privacy_status } : doc;
        })
      );
    }, POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
  }, [documents.map((d) => `${d.processing_status}-${d.privacy_status}`).join(",")]);

  /* Scroll chat to bottom */
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, aiLoading]);

  /* Upload handler */
  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await uploadDocument(file, patientId);
      setToast({ message: `"${file.name}" uploaded successfully.`, variant: "success" });
      fetchDocs();
    } catch (err) {
      setToast({ message: err?.response?.data?.detail || "Upload failed.", variant: "error" });
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  /* AI query handler */
  async function handleAISend(e) {
    e.preventDefault();
    const q = aiQuery.trim();
    if (!q || aiLoading) return;

    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setAiQuery("");
    setAiLoading(true);

    try {
      const res = await searchPatientAI(patientId, q);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: res.answer || "No relevant information found.", sources: res.sources || [] },
      ]);
    } catch (err) {
      const errMsg =
        err?.response?.data?.detail ||
        "I couldn't retrieve an answer. Please ensure this patient's documents have been processed and indexed.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: errMsg, sources: [] },
      ]);
    } finally {
      setAiLoading(false);
    }
  }

  /* ── Demographics panel ── */
  const age = calcAge(patient?.date_of_birth);

  /* ── Render ── */
  if (loadingPatient) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 300 }}>
        <div className="hp-spinner" />
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0, height: "100%" }}>
      {/* ── Breadcrumb ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 20, fontSize: "0.8125rem", color: "var(--hp-text-500)" }}>
        <Link to="/patients" style={{ color: "var(--hp-primary-400)", textDecoration: "none", fontWeight: 500 }}>
          Patients
        </Link>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <span style={{ color: "var(--hp-text-200)", fontWeight: 600 }}>
          {patient ? `${patient.first_name} ${patient.last_name}` : "Workspace"}
        </span>
      </div>

      {/* ── Patient header card ── */}
      <div
        style={{
          background: "linear-gradient(135deg, var(--hp-bg-200) 0%, var(--hp-bg-300) 100%)",
          border: "1px solid var(--hp-border-100)",
          borderRadius: 12,
          padding: "18px 24px",
          marginBottom: 20,
          display: "flex",
          alignItems: "center",
          gap: 20,
        }}
      >
        <div
          style={{
            width: 52,
            height: 52,
            borderRadius: "50%",
            background: "linear-gradient(135deg, var(--hp-primary-700), var(--hp-primary-500))",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "1.25rem",
            fontWeight: 700,
            color: "#fff",
            flexShrink: 0,
          }}
        >
          {patient ? `${patient.first_name?.[0] ?? ""}${patient.last_name?.[0] ?? ""}` : "?"}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
            <h2 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 700, color: "var(--hp-text-100)" }}>
              {patient?.first_name} {patient?.last_name}
            </h2>
            {age !== null && (
              <span style={{ fontSize: "0.8125rem", color: "var(--hp-text-500)", fontWeight: 500 }}>
                {age} yrs
              </span>
            )}
            {patient?.gender && (
              <span
                style={{
                  fontSize: "0.75rem",
                  background: "var(--hp-primary-600)22",
                  color: "var(--hp-primary-400)",
                  padding: "2px 10px",
                  borderRadius: 20,
                  fontWeight: 600,
                }}
              >
                {patient.gender}
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: 20, marginTop: 6, flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.8125rem", color: "var(--hp-text-400)" }}>
              <strong style={{ color: "var(--hp-text-300)" }}>MRN:</strong> {patient?.mrn}
            </span>
            {patient?.date_of_birth && (
              <span style={{ fontSize: "0.8125rem", color: "var(--hp-text-400)" }}>
                <strong style={{ color: "var(--hp-text-300)" }}>DOB:</strong> {fmtDate(patient.date_of_birth)}
              </span>
            )}
            <span style={{ fontSize: "0.8125rem", color: "var(--hp-text-400)" }}>
              <strong style={{ color: "var(--hp-text-300)" }}>Documents:</strong>{" "}
              {loadingDocs ? "…" : documents.length}
            </span>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,.tiff"
          style={{ display: "none" }}
          onChange={handleUpload}
        />
        <button
          id="pw-upload-btn"
          className="hp-btn-primary"
          style={{ width: "auto", whiteSpace: "nowrap", flexShrink: 0 }}
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? (
            <div className="hp-spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          )}
          <span>{uploading ? "Uploading…" : "Upload Document"}</span>
        </button>
      </div>

      {/* ── Split Panel ── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 20,
          flex: 1,
          minHeight: 0,
        }}
        className="pw-split"
      >
        {/* ──── LEFT: Documents ──── */}
        <div
          style={{
            background: "var(--hp-bg-200)",
            border: "1px solid var(--hp-border-100)",
            borderRadius: 12,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "14px 18px",
              borderBottom: "1px solid var(--hp-border-100)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <h3 style={{ margin: 0, fontSize: "0.9375rem", fontWeight: 700, color: "var(--hp-text-100)" }}>
              Clinical Documents
            </h3>
            <span
              style={{
                fontSize: "0.75rem",
                background: "var(--hp-bg-300)",
                color: "var(--hp-text-400)",
                padding: "2px 10px",
                borderRadius: 20,
              }}
            >
              {documents.length} file{documents.length !== 1 ? "s" : ""}
            </span>
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "12px 14px" }}>
            {loadingDocs ? (
              <div style={{ display: "flex", justifyContent: "center", padding: 40 }}>
                <div className="hp-spinner" />
              </div>
            ) : documents.length === 0 ? (
              <div
                style={{
                  textAlign: "center",
                  padding: "48px 24px",
                  color: "var(--hp-text-500)",
                  fontSize: "0.875rem",
                }}
              >
                <svg
                  width="40"
                  height="40"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  style={{ opacity: 0.35, marginBottom: 12 }}
                >
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                <p style={{ margin: "0 0 4px" }}>No documents yet</p>
                <p style={{ margin: 0, fontSize: "0.8125rem", color: "var(--hp-text-600)" }}>
                  Upload a clinical document to get started.
                </p>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {documents.map((doc) => (
                  <div
                    key={doc.document_id}
                    id={`pw-doc-${doc.document_id}`}
                    style={{
                      background: "var(--hp-bg-100)",
                      border: "1px solid var(--hp-border-100)",
                      borderRadius: 8,
                      padding: "10px 14px",
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      transition: "border-color 0.15s",
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.borderColor = "var(--hp-primary-600)")
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.borderColor = "var(--hp-border-100)")
                    }
                    onClick={() => navigate(`/documents/${doc.document_id}`)}
                  >
                    <div
                      style={{
                        width: 34,
                        height: 34,
                        borderRadius: 6,
                        background: "var(--hp-primary-600)1A",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        flexShrink: 0,
                      }}
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="var(--hp-primary-400)"
                        strokeWidth="2"
                      >
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: "0.8125rem",
                          fontWeight: 600,
                          color: "var(--hp-text-100)",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {doc.file_name}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "var(--hp-text-500)", marginTop: 2 }}>
                        {fmtDate(doc.uploaded_at)}
                      </div>
                    </div>
                    <StatusBadge status={doc.processing_status} />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ──── RIGHT: AI Assistant ──── */}
        <div
          style={{
            background: "var(--hp-bg-200)",
            border: "1px solid var(--hp-border-100)",
            borderRadius: 12,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "14px 18px",
              borderBottom: "1px solid var(--hp-border-100)",
              display: "flex",
              alignItems: "center",
              gap: 10,
              background: "linear-gradient(90deg, var(--hp-primary-700)1A, transparent)",
            }}
          >
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: "50%",
                background: "linear-gradient(135deg, var(--hp-primary-600), var(--hp-primary-400))",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5">
                <path d="M12 2a9 9 0 0 1 9 9c0 4.17-2.84 7.67-6.73 8.65L12 22l-2.27-2.35C5.84 18.67 3 15.17 3 11a9 9 0 0 1 9-9z" />
              </svg>
            </div>
            <div>
              <div style={{ fontSize: "0.9375rem", fontWeight: 700, color: "var(--hp-text-100)" }}>
                Clinical AI Assistant
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--hp-text-500)" }}>
                Scoped to this patient's records
              </div>
            </div>
            <div
              style={{
                marginLeft: "auto",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "var(--hp-success)",
                animation: "hp-pulse 2s ease-in-out infinite",
              }}
            />
          </div>

          {/* Messages */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "16px 14px",
              display: "flex",
              flexDirection: "column",
              gap: 14,
            }}
          >
            {messages.map((msg, i) => (
              <AIMessage key={i} role={msg.role} text={msg.text} sources={msg.sources} />
            ))}
            {aiLoading && (
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <div
                  style={{
                    width: 30,
                    height: 30,
                    borderRadius: "50%",
                    background: "linear-gradient(135deg, var(--hp-primary-600), var(--hp-primary-400))",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    color: "#fff",
                    flexShrink: 0,
                  }}
                >
                  AI
                </div>
                <div
                  style={{
                    background: "var(--hp-bg-300)",
                    borderRadius: "4px 12px 12px 12px",
                    padding: "10px 16px",
                    display: "flex",
                    gap: 5,
                    alignItems: "center",
                  }}
                >
                  {[0, 0.18, 0.36].map((delay, k) => (
                    <div
                      key={k}
                      style={{
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        background: "var(--hp-primary-400)",
                        animation: `hp-bounce 1.2s ease-in-out ${delay}s infinite`,
                      }}
                    />
                  ))}
                </div>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>

          {/* Quick suggestions */}
          {messages.length === 1 && !aiLoading && (
            <div style={{ padding: "0 14px 10px", display: "flex", gap: 6, flexWrap: "wrap" }}>
              {[
                "Summarise this patient's history",
                "List current medications",
                "What are the recent lab findings?",
                "Any flagged risk factors?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setAiQuery(q);
                    document.getElementById("pw-ai-input")?.focus();
                  }}
                  style={{
                    fontSize: "0.75rem",
                    padding: "5px 10px",
                    background: "var(--hp-bg-300)",
                    border: "1px solid var(--hp-border-100)",
                    borderRadius: 20,
                    color: "var(--hp-text-300)",
                    cursor: "pointer",
                    transition: "border-color 0.15s, color 0.15s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--hp-primary-400)";
                    e.currentTarget.style.color = "var(--hp-primary-400)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--hp-border-100)";
                    e.currentTarget.style.color = "var(--hp-text-300)";
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <form
            onSubmit={handleAISend}
            style={{
              padding: "12px 14px",
              borderTop: "1px solid var(--hp-border-100)",
              display: "flex",
              gap: 8,
            }}
          >
            <input
              id="pw-ai-input"
              type="text"
              className="hp-input"
              placeholder="Ask a clinical question about this patient…"
              value={aiQuery}
              onChange={(e) => setAiQuery(e.target.value)}
              disabled={aiLoading}
              style={{ flex: 1, fontSize: "0.875rem", padding: "9px 14px" }}
            />
            <button
              type="submit"
              id="pw-ai-send"
              className="hp-btn-primary"
              style={{ width: "auto", padding: "0 14px", flexShrink: 0 }}
              disabled={!aiQuery.trim() || aiLoading}
            >
              {aiLoading ? (
                <div className="hp-spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              )}
            </button>
          </form>
        </div>
      </div>

      <Toast message={toast?.message} variant={toast?.variant} onClose={() => setToast(null)} />

      <style>{`
        @keyframes hp-bounce {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
          40% { transform: scale(1); opacity: 1; }
        }
        @media (max-width: 900px) {
          .pw-split { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
