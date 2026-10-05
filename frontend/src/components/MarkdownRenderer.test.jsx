import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import MarkdownRenderer from "./MarkdownRenderer";

describe("MarkdownRenderer", () => {
  const sampleClinicalMarkdown = `### Summary of Patient History

#### Document: David Lee.pdf (Chunk #0)
- **Patient Name:** David Lee
- **Gender:** Male
- **Date of Visit:** [DATE]
- **Reason for Visit:** Sudden onset of severe headache.
- **Procedure:** CT Head Without Contrast

#### Findings:
- No evidence of acute territorial infarct.
- No abnormal densities suggesting acute intracranial hemorrhage.

**Impression:** Negative for acute intracranial hemorrhage.`;

  it("1. renders Markdown headings as proper heading elements", () => {
    render(<MarkdownRenderer content={sampleClinicalMarkdown} />);

    // Check h3 heading
    const h3 = screen.getByRole("heading", { level: 3, name: /Summary of Patient History/i });
    expect(h3).toBeDefined();
    expect(h3.tagName).toBe("H3");

    // Check h4 headings
    const h4Elements = screen.getAllByRole("heading", { level: 4 });
    expect(h4Elements.length).toBeGreaterThanOrEqual(2);
    expect(h4Elements[0].textContent).toContain("Document: David Lee.pdf (Chunk #0)");
    expect(h4Elements[1].textContent).toContain("Findings:");
  });

  it("2. renders Markdown bullets as bullet lists", () => {
    const { container } = render(<MarkdownRenderer content={sampleClinicalMarkdown} />);

    // Must render ul and li elements
    const lists = container.querySelectorAll("ul");
    expect(lists.length).toBeGreaterThanOrEqual(2);

    const listItems = container.querySelectorAll("li");
    expect(listItems.length).toBeGreaterThanOrEqual(7);

    // Verify content inside list items
    const itemTexts = Array.from(listItems).map((li) => li.textContent);
    expect(itemTexts.some((t) => t.includes("Patient Name: David Lee"))).toBe(true);
    expect(itemTexts.some((t) => t.includes("No evidence of acute territorial infarct."))).toBe(true);
  });

  it("3. renders bold Markdown as bold text elements", () => {
    const { container } = render(<MarkdownRenderer content={sampleClinicalMarkdown} />);

    const strongElements = container.querySelectorAll("strong");
    expect(strongElements.length).toBeGreaterThanOrEqual(6);

    const strongTexts = Array.from(strongElements).map((el) => el.textContent);
    expect(strongTexts).toContain("Patient Name:");
    expect(strongTexts).toContain("Gender:");
    expect(strongTexts).toContain("Impression:");
  });

  it("4. does not display raw ###, ####, -, and ** as literal formatting characters", () => {
    const { container } = render(<MarkdownRenderer content={sampleClinicalMarkdown} />);
    const textContent = container.textContent;

    // The text content should not contain markdown syntax markers
    expect(textContent).not.toContain("###");
    expect(textContent).not.toContain("####");
    expect(textContent).not.toContain("**Patient Name:**");
    expect(textContent).not.toContain("**Impression:**");
  });

  it("supports code blocks, numbered lists, and italics", () => {
    const markdown = `1. First step\n2. Second step\n\n*Important note*\n\n\`\`\`json\n{"status": "ok"}\n\`\`\``;
    const { container } = render(<MarkdownRenderer content={markdown} />);

    const ol = container.querySelector("ol");
    expect(ol).toBeDefined();
    expect(container.querySelectorAll("ol > li").length).toBe(2);

    const em = container.querySelector("em");
    expect(em).toBeDefined();
    expect(em.textContent).toBe("Important note");

    const code = container.querySelector("code");
    expect(code).toBeDefined();
    expect(code.textContent).toContain('"status": "ok"');
  });
});
