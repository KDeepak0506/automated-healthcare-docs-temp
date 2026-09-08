import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, Link } from "react-router-dom";
import { listDocuments, getDocumentStatus, deleteDocument } from "../api/documents";
import DocumentList from "../components/DocumentList";
import Toast from "../components/Toast";

const POLL_INTERVAL_MS = 4000;
const ACTIVE_STATUSES = ["Pending", "Processing"];

export default function DocumentsPage() {
  const location = useLocation();
  const newestId = location.state?.newestId || null;

  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null);
  
  // Search & Filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [processingFilter, setProcessingFilter] = useState("");
  const [privacyFilter, setPrivacyFilter] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [paginationMeta, setPaginationMeta] = useState({ total: 0, total_pages: 1 });

  const pollRef = useRef(null);

  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const params = {
        page,
        page_size: pageSize,
      };
      if (searchQuery.trim()) params.search = searchQuery.trim();
      if (processingFilter) params.processing_status = processingFilter;
      if (privacyFilter) params.privacy_status = privacyFilter;

      const data = await listDocuments(params);
      if (data && data.items) {
        setDocuments(data.items);
        setPaginationMeta({
          total: data.total,
          total_pages: data.total_pages,
        });
      } else {
        setDocuments(Array.isArray(data) ? data : []);
        setPaginationMeta({ total: Array.isArray(data) ? data.length : 0, total_pages: 1 });
      }
    } catch (err) {
      setToast({ message: err.message || "Couldn't load documents.", variant: "error" });
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, searchQuery, processingFilter, privacyFilter]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Handle document deletion
  const handleDeleteDocument = async (documentId) => {
    try {
      await deleteDocument(documentId);
      setToast({ message: "Document deleted successfully.", variant: "success" });
      fetchDocuments();
    } catch (err) {
      setToast({ message: err.response?.data?.detail || "Failed to delete document.", variant: "error" });
    }
  };

  // Poll status for active documents
  useEffect(() => {
    const activeDocs = documents.filter((d) =>
      ACTIVE_STATUSES.includes(d.processing_status) ||
      (d.privacy_status && ["pending", "processing"].includes(d.privacy_status))
    );

    if (activeDocs.length === 0) {
      clearInterval(pollRef.current);
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const updates = await Promise.all(
          activeDocs.map((d) => getDocumentStatus(d.document_id).catch(() => null))
        );
        setDocuments((prev) =>
          prev.map((doc) => {
            const update = updates.find((u) => u && u.document_id === doc.document_id);
            return update
              ? {
                  ...doc,
                  processing_status: update.status,
                  privacy_status: update.privacy_status,
                }
              : doc;
          })
        );
      } catch {
        // silent retry
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(pollRef.current);
  }, [documents.map((d) => `${d.processing_status}-${d.privacy_status}`).join(",")]);

  return (
    <div>
      <div className="hp-section-header" style={{ marginBottom: 20 }}>
        <div>
          <h1 className="hp-page-title" style={{ fontSize: "1.5rem" }}>Healthcare Document Repository</h1>
          <p style={{ color: "var(--hp-text-500)", fontSize: "0.875rem", margin: "4px 0 0" }}>
            Total registered documents: {paginationMeta.total}
          </p>
        </div>
        <Link to="/upload" className="hp-btn-primary" style={{ width: "auto" }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
          <span>Upload Document</span>
        </Link>
      </div>

      {/* Search & Filter Toolbar */}
      <div
        style={{
          display: "flex",
          gap: "12px",
          marginBottom: "20px",
          flexWrap: "wrap",
          alignItems: "center",
          background: "#ffffff",
          padding: "14px 18px",
          borderRadius: "10px",
          border: "1px solid var(--hp-border, #e2e8f0)",
        }}
      >
        <div style={{ flex: 1, minWidth: "220px", display: "flex", alignItems: "center", gap: "8px" }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <input
            type="text"
            placeholder="Search by filename..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setPage(1);
            }}
            style={{
              width: "100%",
              border: "none",
              outline: "none",
              fontSize: "0.875rem",
              color: "#0f172a",
            }}
          />
        </div>

        <select
          value={processingFilter}
          onChange={(e) => {
            setProcessingFilter(e.target.value);
            setPage(1);
          }}
          style={{
            padding: "6px 12px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "0.8125rem",
            color: "#334155",
            background: "#ffffff",
          }}
        >
          <option value="">All OCR Statuses</option>
          <option value="Pending">Pending</option>
          <option value="Processing">Processing</option>
          <option value="Completed">Completed</option>
          <option value="Failed">Failed</option>
        </select>

        <select
          value={privacyFilter}
          onChange={(e) => {
            setPrivacyFilter(e.target.value);
            setPage(1);
          }}
          style={{
            padding: "6px 12px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "0.8125rem",
            color: "#334155",
            background: "#ffffff",
          }}
        >
          <option value="">All Privacy Statuses</option>
          <option value="pending">Pending</option>
          <option value="processing">Processing</option>
          <option value="completed">Protected</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      <DocumentList
        documents={documents}
        loading={loading}
        newestId={newestId}
        onDocumentUpdated={(updatedDoc) => {
          setDocuments((prev) =>
            prev.map((d) => (d.document_id === updatedDoc.document_id ? { ...d, ...updatedDoc } : d))
          );
        }}
        onDelete={handleDeleteDocument}
      />

      {/* Pagination Controls */}
      {paginationMeta.total_pages > 1 && (
        <div
          style={{
            display: "flex",
            justify: "space-between",
            alignItems: "center",
            marginTop: "16px",
            padding: "12px 16px",
            background: "#ffffff",
            borderRadius: "8px",
            border: "1px solid #e2e8f0",
          }}
        >
          <span style={{ fontSize: "0.8125rem", color: "#64748b" }}>
            Page {page} of {paginationMeta.total_pages} ({paginationMeta.total} items total)
          </span>
          <div style={{ display: "flex", gap: "8px" }}>
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              style={{
                padding: "6px 14px",
                borderRadius: "6px",
                border: "1px solid #cbd5e1",
                background: page <= 1 ? "#f1f5f9" : "#ffffff",
                color: page <= 1 ? "#94a3b8" : "#334155",
                fontSize: "0.8125rem",
                cursor: page <= 1 ? "not-allowed" : "pointer",
                fontWeight: 500,
              }}
            >
              Previous
            </button>
            <button
              onClick={() => setPage((p) => Math.min(paginationMeta.total_pages, p + 1))}
              disabled={page >= paginationMeta.total_pages}
              style={{
                padding: "6px 14px",
                borderRadius: "6px",
                border: "1px solid #cbd5e1",
                background: page >= paginationMeta.total_pages ? "#f1f5f9" : "#ffffff",
                color: page >= paginationMeta.total_pages ? "#94a3b8" : "#334155",
                fontSize: "0.8125rem",
                cursor: page >= paginationMeta.total_pages ? "not-allowed" : "pointer",
                fontWeight: 500,
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}

      <Toast
        message={toast?.message}
        variant={toast?.variant}
        onClose={() => setToast(null)}
      />
    </div>
  );
}
