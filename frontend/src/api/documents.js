import client from "./client";

// POST /api/v1/documents
export async function uploadDocument(file, patientId) {
  const form = new FormData();
  form.append("file", file);
  if (patientId) form.append("patient_id", patientId);

  const { data } = await client.post("/documents", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data; // { document_id, status, uploaded_at }
}

// GET /api/v1/documents — backend returns a raw array, no pagination yet.
export async function listDocuments() {
  const { data } = await client.get("/documents");
  return data; // DocumentResponse[]
}

export async function getDocumentStatus(documentId) {
  const { data } = await client.get(`/documents/${documentId}`);
  return {
    document_id: data.document_id,
    status: data.processing_status,
    privacy_status: data.privacy_status,
  };
}

export async function getDocumentSanitizedText(documentId) {
  const { data } = await client.get(`/documents/${documentId}/sanitized-text`);
  return data; // { document_id, privacy_status, sanitized_text }
}

export async function getDocumentEntities(documentId) {
  const { data } = await client.get(`/documents/${documentId}/entities`);
  return data; // { document_id, entities: [...], total_count }
}

export async function classifyDocument(documentId) {
  const { data } = await client.post(`/documents/${documentId}/classify`);
  return data; // { document_id, document_type, classification_confidence, cached, truncated }
}

export async function summarizeDocument(documentId) {
  const { data } = await client.post(`/documents/${documentId}/summarize`);
  return data; // { document_id, summary, key_findings, cached, truncated }
}

