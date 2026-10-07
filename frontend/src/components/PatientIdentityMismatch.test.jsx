import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import UploadForm from "./UploadForm";
import * as documentsApi from "../api/documents";

describe("Patient Identity Mismatch UI Handling", () => {
  it("5. displays patient identity mismatch error clearly and preserves upload usability", async () => {
    // Mock uploadDocument to reject with PATIENT_IDENTITY_MISMATCH
    vi.spyOn(documentsApi, "uploadDocument").mockRejectedValue({
      status: 400,
      code: "PATIENT_IDENTITY_MISMATCH",
      message: "Document appears to belong to John Smith, but this workspace is for David Lee. The document was not uploaded.",
      detected_patient_name: "John Smith",
      target_patient_name: "David Lee",
    });

    const onUploaded = vi.fn();
    const onError = vi.fn();

    render(
      <UploadForm
        patientId="david-lee-id"
        patientName="David Lee"
        onUploaded={onUploaded}
        onError={onError}
      />
    );

    // Verify patient context banner is shown
    expect(screen.getByText("David Lee")).toBeDefined();
    expect(screen.getByText(/Uploading for patient:/i)).toBeDefined();

    // Select a file
    const file = new File(["dummy content"], "John_Smith.pdf", { type: "application/pdf" });
    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [file] } });

    // Verify file is selected
    expect(screen.getByText("John_Smith.pdf")).toBeDefined();

    // Click upload
    const uploadBtn = screen.getByRole("button", { name: /Upload Document/i });
    fireEvent.click(uploadBtn);

    // Verify clear mismatch error banner appears
    await waitFor(() => {
      const alert = screen.getByRole("alert");
      expect(alert).toBeDefined();
      expect(alert.textContent).toContain("John Smith");
      expect(alert.textContent).toContain("David Lee");
      expect(alert.textContent).toContain("Please verify the patient before uploading");
    });

    // Verify onError callback was called with the message
    expect(onError).toHaveBeenCalled();

    // Verify upload UI remains fully usable after rejection so user can pick the right file
    expect(uploadBtn).toBeDefined();
    expect(uploadBtn.disabled).toBe(false);

    // User can remove file or select another
    const removeBtn = screen.getByRole("button", { name: /Remove file/i });
    fireEvent.click(removeBtn);

    expect(screen.queryByText("John_Smith.pdf")).toBeNull();
  });
});
