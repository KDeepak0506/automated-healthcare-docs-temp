import client from "./client";

export async function listPatients(params = {}) {
  const response = await client.get("/patients", { params });
  return response.data;
}

export async function getPatient(patientId) {
  const response = await client.get(`/patients/${patientId}`);
  return response.data;
}

export async function createPatient(data) {
  const response = await client.post("/patients", data);
  return response.data;
}

export async function updatePatient(patientId, data) {
  const response = await client.put(`/patients/${patientId}`, data);
  return response.data;
}

export async function assignPatient(patientId, userId) {
  const response = await client.post(`/patients/${patientId}/assign`, { user_id: userId });
  return response.data;
}

export async function searchPatientAI(patientId, query, topK = 5) {
  const response = await client.post(`/patients/${patientId}/search`, {
    query,
    top_k: topK,
  });
  return response.data;
}
