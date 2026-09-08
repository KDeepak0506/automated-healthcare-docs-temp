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

// GET /api/v1/documents — paginated, searchable, filterable (M7)
export async function listDocuments(params = {}) {
  const { data } = await client.get("/documents", { params });
  return data; // { items: DocumentResponse[], page, page_size, total, total_pages }
}

// DELETE /api/v1/documents/{documentId} (M7)
export async function deleteDocument(documentId) {
  const { data } = await client.delete(`/documents/${documentId}`);
  return data;
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

export async function indexDocument(documentId) {
  const { data } = await client.post(`/documents/${documentId}/index`);
  return data; // { document_id, chunks_created, status }
}

export async function searchDocument(documentId, query, topK = 5) {
  const { data } = await client.post(`/documents/${documentId}/search`, {
    query,
    top_k: topK,
  });
  return data; // { answer, sources }
}

// GET /api/v1/documents/{documentId}/sources/{chunkId} (M8)
export async function getChunkSource(documentId, chunkId) {
  const { data } = await client.get(`/documents/${documentId}/sources/${chunkId}`);
  return data; // { document_id, chunk_id, chunk_index, page_number, similarity_score, text, start_offset, end_offset }
}
