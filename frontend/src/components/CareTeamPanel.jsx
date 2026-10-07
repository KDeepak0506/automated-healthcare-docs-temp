import { useCallback, useEffect, useState } from "react";
import {
  assignPatient,
  listAssignments,
  listUsers,
  unassignPatient,
} from "../api/patients";

export default function CareTeamPanel({ patientId }) {
  const [assignments, setAssignments] = useState([]);
  const [nurses, setNurses] = useState([]);
  const [selectedNurseId, setSelectedNurseId] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [hidden, setHidden] = useState(false);

  const fetchData = useCallback(async () => {
    if (!patientId) return;
    try {
      setLoading(true);
      setErrorMsg(null);
      const [assignmentsRes, nursesRes] = await Promise.all([
        listAssignments(patientId),
        listUsers("nurse"),
      ]);
      setAssignments(Array.isArray(assignmentsRes) ? assignmentsRes : []);
      setNurses(Array.isArray(nursesRes) ? nursesRes : []);
      setHidden(false);
    } catch (err) {
      if (err?.status === 403) {
        setHidden(true);
        return;
      }
      setErrorMsg(err?.message || "Failed to load care team information.");
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (hidden) {
    return null;
  }

  // Nurses not yet assigned
  const assignedUserIds = new Set(
    assignments.map((a) => a.user_id || a.user?.user_id)
  );
  const availableNurses = nurses.filter(
    (nurse) => !assignedUserIds.has(nurse.user_id)
  );

  async function handleAssign(e) {
    e.preventDefault();
    if (!selectedNurseId || actionLoading) return;

    try {
      setActionLoading(true);
      setErrorMsg(null);
      await assignPatient(patientId, selectedNurseId);
      setSelectedNurseId("");
      await fetchData();
    } catch (err) {
      if (err?.status === 403) {
        setHidden(true);
        return;
      }
      setErrorMsg(err?.message || "Failed to assign nurse.");
    } finally {
      setActionLoading(false);
    }
  }

  async function handleRemove(userId) {
    if (!userId || actionLoading) return;

    try {
      setActionLoading(true);
      setErrorMsg(null);
      await unassignPatient(patientId, userId);
      await fetchData();
    } catch (err) {
      if (err?.status === 403) {
        setHidden(true);
        return;
      }
      setErrorMsg(err?.message || "Failed to unassign member.");
    } finally {
      setActionLoading(false);
    }
  }

  return (
    <div
      id="care-team-panel"
      className="hp-card"
      style={{
        marginBottom: 20,
        padding: "18px 24px",
        background: "var(--hp-surface)",
        border: "1px solid var(--hp-border-100)",
        borderRadius: 12,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 16,
          flexWrap: "wrap",
          gap: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: "50%",
              background: "var(--hp-primary-light)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--hp-primary)",
            }}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
          </div>
          <div>
            <h3
              style={{
                margin: 0,
                fontSize: "1rem",
                fontWeight: 600,
                color: "var(--hp-text-900)",
              }}
            >
              Care Team Assignment
            </h3>
            <p
              style={{
                margin: "2px 0 0 0",
                fontSize: "0.8125rem",
                color: "var(--hp-text-500)",
              }}
            >
              Manage assigned nurses with access to this patient record.
            </p>
          </div>
        </div>

        {/* Assign Nurse Form */}
        <form
          onSubmit={handleAssign}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
          }}
        >
          <select
            id="nurse-select"
            className="hp-select"
            value={selectedNurseId}
            onChange={(e) => setSelectedNurseId(e.target.value)}
            disabled={actionLoading || loading || availableNurses.length === 0}
            style={{
              minWidth: 200,
              padding: "7px 12px",
              fontSize: "0.875rem",
            }}
          >
            <option value="">
              {availableNurses.length === 0
                ? "No available nurses"
                : "Select a nurse..."}
            </option>
            {availableNurses.map((nurse) => (
              <option key={nurse.user_id} value={nurse.user_id}>
                {nurse.name} ({nurse.email})
              </option>
            ))}
          </select>

          <button
            type="submit"
            id="assign-nurse-btn"
            className="hp-btn-primary"
            disabled={
              !selectedNurseId || actionLoading || loading
            }
            style={{
              width: "auto",
              padding: "7px 16px",
              fontSize: "0.875rem",
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            {actionLoading ? (
              <div
                className="hp-spinner"
                style={{ width: 14, height: 14, borderWidth: 2 }}
              />
            ) : (
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            )}
            <span>Assign</span>
          </button>
        </form>
      </div>

      {errorMsg && (
        <div
          className="hp-error-banner"
          style={{
            marginBottom: 12,
            padding: "8px 12px",
            fontSize: "0.8125rem",
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{errorMsg}</span>
        </div>
      )}

      {loading ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "20px 0",
          }}
        >
          <div className="hp-spinner" style={{ width: 24, height: 24 }} />
        </div>
      ) : assignments.length === 0 ? (
        <div
          style={{
            padding: "16px",
            background: "var(--hp-bg)",
            borderRadius: 8,
            color: "var(--hp-text-500)",
            fontSize: "0.875rem",
            textAlign: "center",
          }}
        >
          No care team members assigned yet. Use the dropdown above to assign a nurse.
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: 12,
          }}
        >
          {assignments.map((assignment) => {
            const memberId = assignment.user_id || assignment.user?.user_id;
            const memberName =
              assignment.name || assignment.user?.name || "Care Team Member";
            const memberEmail = assignment.email || assignment.user?.email || "";
            const memberRole =
              assignment.role || assignment.user?.role || "nurse";

            return (
              <div
                key={assignment.assignment_id || memberId}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "10px 14px",
                  background: "var(--hp-bg)",
                  border: "1px solid var(--hp-border-100)",
                  borderRadius: 8,
                  gap: 10,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    minWidth: 0,
                  }}
                >
                  <div
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: "50%",
                      background: "linear-gradient(135deg, var(--hp-primary), var(--hp-primary-hover))",
                      color: "#fff",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "0.875rem",
                      fontWeight: 600,
                      flexShrink: 0,
                    }}
                  >
                    {memberName.charAt(0).toUpperCase()}
                  </div>
                  <div style={{ minWidth: 0, overflow: "hidden" }}>
                    <div
                      style={{
                        fontSize: "0.875rem",
                        fontWeight: 600,
                        color: "var(--hp-text-900)",
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                      title={memberName}
                    >
                      {memberName}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        marginTop: 2,
                      }}
                    >
                      <span
                        style={{
                          fontSize: "0.6875rem",
                          textTransform: "uppercase",
                          letterSpacing: "0.04em",
                          fontWeight: 600,
                          padding: "1px 6px",
                          borderRadius: 4,
                          background:
                            memberRole === "nurse"
                              ? "var(--hp-primary-light)"
                              : "var(--hp-info-light)",
                          color:
                            memberRole === "nurse"
                              ? "var(--hp-primary)"
                              : "var(--hp-info)",
                        }}
                      >
                        {memberRole}
                      </span>
                      {memberEmail && (
                        <span
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--hp-text-400)",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                          title={memberEmail}
                        >
                          {memberEmail}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  id={`remove-team-btn-${memberId}`}
                  className="hp-btn-secondary"
                  onClick={() => handleRemove(memberId)}
                  disabled={actionLoading}
                  style={{
                    padding: "4px 10px",
                    fontSize: "0.75rem",
                    color: "var(--hp-danger)",
                    borderColor: "rgba(201, 42, 42, 0.3)",
                    background: "var(--hp-surface)",
                    flexShrink: 0,
                  }}
                  title="Remove assignment"
                >
                  Remove
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
