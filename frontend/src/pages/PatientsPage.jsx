import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { listPatients, createPatient } from "../api/patients";
import Toast from "../components/Toast";

export default function PatientsPage() {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [toast, setToast] = useState(null);

  // Modal state for creating new patient
  const [showModal, setShowModal] = useState(false);
  const [mrn, setMrn] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [dob, setDob] = useState("");
  const [gender, setGender] = useState("Unspecified");
  const [submitting, setSubmitting] = useState(false);

  const navigate = useNavigate();

  const fetchPatients = useCallback(async () => {
    try {
      setLoading(true);
      const params = { page, page_size: 10 };
      if (search.trim()) params.search = search.trim();
      const data = await listPatients(params);
      setPatients(data.items || []);
      setTotal(data.total || 0);
      setTotalPages(data.total_pages || 1);
    } catch (err) {
      setToast({ message: err.message || "Failed to load patients.", variant: "error" });
    } finally {
      setLoading(false);
    }
  }, [page, search]);

  useEffect(() => {
    fetchPatients();
  }, [fetchPatients]);

  const handleCreatePatient = async (e) => {
    e.preventDefault();
    if (!mrn.trim() || !firstName.trim() || !lastName.trim()) {
      setToast({ message: "MRN, First Name, and Last Name are required.", variant: "error" });
      return;
    }
    try {
      setSubmitting(true);
      const newPatient = await createPatient({
        mrn: mrn.trim(),
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        date_of_birth: dob || null,
        gender: gender || null,
      });
      setToast({ message: `Patient ${newPatient.first_name} ${newPatient.last_name} created successfully!`, variant: "success" });
      setShowModal(false);
      setMrn("");
      setFirstName("");
      setLastName("");
      setDob("");
      setGender("Unspecified");
      fetchPatients();
      navigate(`/patients/${newPatient.patient_id}`);
    } catch (err) {
      setToast({ message: err.response?.data?.detail || err.message || "Failed to create patient.", variant: "error" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
      {/* Header */}
      <div className="hp-section-header" style={{ marginBottom: 24, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="hp-page-title" style={{ fontSize: "1.5rem" }}>Patient Workspaces</h1>
          <p style={{ color: "var(--hp-text-500)", fontSize: "0.875rem", margin: "4px 0 0" }}>
            Select or register a patient to open their clinical document workspace ({total} accessible patients)
          </p>
        </div>
        <button
          className="hp-btn-primary"
          onClick={() => setShowModal(true)}
          style={{ width: "auto" }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          <span>Register New Patient</span>
        </button>
      </div>

      {/* Toolbar */}
      <div
        style={{
          display: "flex",
          gap: "12px",
          marginBottom: "24px",
          background: "#ffffff",
          padding: "12px 18px",
          borderRadius: "10px",
          border: "1px solid var(--hp-border, #e2e8f0)",
          alignItems: "center",
        }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="2">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <input
          type="text"
          placeholder="Search by MRN, First or Last Name..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          style={{
            flex: 1,
            border: "none",
            outline: "none",
            fontSize: "0.9375rem",
            color: "#0f172a",
          }}
        />
      </div>

      {/* Patient List */}
      {loading ? (
        <div style={{ padding: "40px", textAlign: "center", color: "#64748b" }}>
          Loading accessible patients...
        </div>
      ) : patients.length === 0 ? (
        <div
          style={{
            background: "#ffffff",
            padding: "48px 24px",
            borderRadius: "12px",
            border: "1px dashed #cbd5e1",
            textAlign: "center",
          }}
        >
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5" style={{ marginBottom: 12 }}>
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
            <circle cx="9" cy="7" r="4"></circle>
            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
          </svg>
          <h3 style={{ margin: "0 0 8px", color: "#1e293b", fontSize: "1.125rem" }}>No Patients Found</h3>
          <p style={{ margin: "0 0 20px", color: "#64748b", fontSize: "0.875rem" }}>
            {search ? "No patient matches your search criteria." : "Register your first patient to begin managing patient documents."}
          </p>
          <button className="hp-btn-primary" onClick={() => setShowModal(true)} style={{ width: "auto" }}>
            Register New Patient
          </button>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "16px" }}>
          {patients.map((p) => (
            <div
              key={p.patient_id}
              style={{
                background: "#ffffff",
                borderRadius: "12px",
                border: "1px solid #e2e8f0",
                padding: "20px",
                display: "flex",
                flexDirection: "column",
                justifySpace: "space-between",
                boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
                transition: "box-shadow 0.2s ease, border-color 0.2s ease",
              }}
            >
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: "1.125rem", color: "#0f172a", fontWeight: 600 }}>
                      {p.first_name} {p.last_name}
                    </h3>
                    <span style={{ fontSize: "0.8125rem", color: "#0284c7", fontWeight: 600, background: "#e0f2fe", padding: "2px 8px", borderRadius: "4px", marginTop: 4, display: "inline-block" }}>
                      MRN: {p.mrn}
                    </span>
                  </div>
                  <span style={{ fontSize: "0.75rem", background: "#f1f5f9", color: "#475569", padding: "4px 8px", borderRadius: "6px", fontWeight: 500 }}>
                    {p.document_count} {p.document_count === 1 ? "Doc" : "Docs"}
                  </span>
                </div>

                <div style={{ fontSize: "0.8125rem", color: "#64748b", display: "flex", flexDirection: "column", gap: 4, marginBottom: 16 }}>
                  {p.gender && <div><strong>Gender:</strong> {p.gender}</div>}
                  {p.date_of_birth && <div><strong>DOB:</strong> {p.date_of_birth}</div>}
                  <div><strong>Added:</strong> {new Date(p.created_at).toLocaleDateString()}</div>
                </div>
              </div>

              <Link
                to={`/patients/${p.patient_id}`}
                className="hp-btn-primary"
                style={{ textDecoration: "none", textAlign: "center", display: "flex", justifyContent: "center", alignItems: "center", gap: 8, padding: "8px 14px", fontSize: "0.875rem" }}
              >
                <span>Open Patient Workspace</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
              </Link>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "24px", padding: "12px 16px", background: "#ffffff", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
          <span style={{ fontSize: "0.8125rem", color: "#64748b" }}>
            Page {page} of {totalPages} ({total} patients total)
          </span>
          <div style={{ display: "flex", gap: "8px" }}>
            <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1} className="hp-btn-secondary" style={{ padding: "6px 12px", fontSize: "0.8125rem" }}>
              Previous
            </button>
            <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page >= totalPages} className="hp-btn-secondary" style={{ padding: "6px 12px", fontSize: "0.8125rem" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Create Patient Modal */}
      {showModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(15, 23, 42, 0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 16 }}>
          <div style={{ background: "#ffffff", borderRadius: "12px", width: "100%", maxWidth: "480px", padding: "24px", boxShadow: "0 20px 25px -5px rgba(0,0,0,0.1)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h2 style={{ margin: 0, fontSize: "1.25rem", color: "#0f172a" }}>Register New Patient</h2>
              <button onClick={() => setShowModal(false)} style={{ border: "none", background: "transparent", cursor: "pointer", color: "#64748b" }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>

            <form onSubmit={handleCreatePatient}>
              <div className="hp-field">
                <label htmlFor="mrn">Medical Record Number (MRN) *</label>
                <input id="mrn" className="hp-input" type="text" required placeholder="e.g. PAT-10492" value={mrn} onChange={(e) => setMrn(e.target.value)} />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div className="hp-field">
                  <label htmlFor="firstName">First Name *</label>
                  <input id="firstName" className="hp-input" type="text" required placeholder="John" value={firstName} onChange={(e) => setFirstName(e.target.value)} />
                </div>
                <div className="hp-field">
                  <label htmlFor="lastName">Last Name *</label>
                  <input id="lastName" className="hp-input" type="text" required placeholder="Doe" value={lastName} onChange={(e) => setLastName(e.target.value)} />
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div className="hp-field">
                  <label htmlFor="dob">Date of Birth</label>
                  <input id="dob" className="hp-input" type="date" value={dob} onChange={(e) => setDob(e.target.value)} />
                </div>
                <div className="hp-field">
                  <label htmlFor="gender">Gender</label>
                  <select id="gender" className="hp-select" value={gender} onChange={(e) => setGender(e.target.value)}>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                    <option value="Unspecified">Unspecified</option>
                  </select>
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px", marginTop: 24 }}>
                <button type="button" className="hp-btn-secondary" onClick={() => setShowModal(false)} style={{ width: "auto" }}>
                  Cancel
                </button>
                <button type="submit" className="hp-btn-primary" disabled={submitting} style={{ width: "auto" }}>
                  {submitting ? "Saving..." : "Register Patient"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <Toast message={toast?.message} variant={toast?.variant} onClose={() => setToast(null)} />
    </div>
  );
}
