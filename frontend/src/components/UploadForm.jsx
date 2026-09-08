import { useRef, useState } from "react";
import { uploadDocument } from "../api/documents";

const ACCEPTED_TYPES = ["application/pdf", "image/jpeg", "image/jpg", "image/png"];
const ACCEPTED_EXT = [".pdf", ".jpg", ".jpeg", ".png"];
const MAX_SIZE_MB = 20;

function validateFile(file) {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return "Unsupported file type. Upload a PDF, JPG, JPEG, or PNG.";
  }
  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    return `File exceeds the ${MAX_SIZE_MB}MB limit.`;
  }
  return null;
}

export default function UploadForm({ onUploaded, onError, patientId = null, patientName = null }) {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef(null);

  function handleFile(selected) {
    setError(null);
    const validationError = validateFile(selected);
    if (validationError) {
      setError(validationError);
      setFile(null);
      return;
    }
    setFile(selected);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFile(dropped);
  }

  async function handleUpload() {
    if (!file) {
      setError("Please select a file to upload.");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const result = await uploadDocument(file, patientId);
      onUploaded?.(result);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (err) {
      const msg = err.message || "Upload failed. Please try again.";
      setError(msg);
      onError?.(msg);
    } finally {
      setUploading(false);
    }
  }

  const getFileExtension = (filename) => {
    return filename.split(".").pop().toUpperCase();
  };

  return (
    <div className="hp-upload-workspace">
      <div className="hp-section-header">
        <h2 className="hp-section-title">Upload Healthcare Document</h2>
        <span className="hp-dropzone-subtitle" style={{ margin: 0 }}>Max file size: {MAX_SIZE_MB}MB</span>
      </div>

      {/* Patient context banner — only shown when uploading from a patient workspace */}
      {patientId && patientName && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            background: "var(--hp-primary-light, #E6F4F4)",
            border: "1px solid var(--hp-primary-600, #0A5C5F)44",
            borderRadius: 8,
            padding: "10px 16px",
            marginBottom: 16,
            fontSize: "0.875rem",
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--hp-primary, #0D7377)" strokeWidth="2">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
            <circle cx="9" cy="7" r="4" />
          </svg>
          <span style={{ color: "var(--hp-primary-hover, #0A5C5F)", fontWeight: 600 }}>
            Uploading for patient:&nbsp;
          </span>
          <span style={{ color: "var(--hp-text-900, #1A1D21)", fontWeight: 700 }}>
            {patientName}
          </span>
          <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "var(--hp-text-500, #6C757D)" }}>
            Document will be automatically associated with this patient.
          </span>
        </div>
      )}

      {error && (
        <div className="hp-error-banner" role="alert">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>{error}</span>
        </div>
      )}

      <label
        className={`hp-dropzone ${dragActive ? "active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXT.join(",")}
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        <div className="hp-upload-cloud-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
        </div>
        <div className="hp-dropzone-title">
          {dragActive ? "Drop document to select" : "Drag and drop your document here"}
        </div>
        <div className="hp-dropzone-subtitle">or click to browse files from your computer</div>
        
        <div className="hp-format-pills">
          <span className="hp-format-pill">PDF</span>
          <span className="hp-format-pill">JPG</span>
          <span className="hp-format-pill">JPEG</span>
          <span className="hp-format-pill">PNG</span>
        </div>
      </label>

      {file && (
        <div className="hp-selected-file-card">
          <div className="hp-file-details">
            <div className="hp-file-icon">{getFileExtension(file.name)}</div>
            <div>
              <div className="hp-file-name">{file.name}</div>
              <div className="hp-file-meta">{(file.size / (1024 * 1024)).toFixed(2)} MB • Ready for processing</div>
            </div>
          </div>
          <button
            className="hp-remove-file-btn"
            onClick={() => setFile(null)}
            disabled={uploading}
            type="button"
          >
            Remove file
          </button>
        </div>
      )}

      <div className="hp-upload-actions">
        <button
          className="hp-btn-primary"
          onClick={handleUpload}
          disabled={uploading}
          type="button"
        >
          {uploading ? (
            <>
              <span className="hp-spinner" />
              <span>Uploading &amp; Processing...</span>
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              <span>Upload Document</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
