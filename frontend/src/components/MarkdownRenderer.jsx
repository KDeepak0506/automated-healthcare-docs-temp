import React from "react";
import ReactMarkdown from "react-markdown";
import "./MarkdownRenderer.css";

export default function MarkdownRenderer({ content, className = "" }) {
  if (!content) return null;

  return (
    <div className={`hp-markdown-container ${className}`}>
      <ReactMarkdown
        components={{
          h1: ({ node, ...props }) => <h1 className="hp-md-heading hp-md-h1" {...props} />,
          h2: ({ node, ...props }) => <h2 className="hp-md-heading hp-md-h2" {...props} />,
          h3: ({ node, ...props }) => <h3 className="hp-md-heading hp-md-h3" {...props} />,
          h4: ({ node, ...props }) => <h4 className="hp-md-heading hp-md-h4" {...props} />,
          h5: ({ node, ...props }) => <h5 className="hp-md-heading hp-md-h5" {...props} />,
          h6: ({ node, ...props }) => <h6 className="hp-md-heading hp-md-h6" {...props} />,
          p: ({ node, ...props }) => <p className="hp-md-paragraph" {...props} />,
          ul: ({ node, ...props }) => <ul className="hp-md-list hp-md-ul" {...props} />,
          ol: ({ node, ...props }) => <ol className="hp-md-list hp-md-ol" {...props} />,
          li: ({ node, ...props }) => <li className="hp-md-list-item" {...props} />,
          strong: ({ node, ...props }) => <strong className="hp-md-strong" {...props} />,
          em: ({ node, ...props }) => <em className="hp-md-em" {...props} />,
          code: ({ node, inline, ...props }) =>
            inline ? (
              <code className="hp-md-inline-code" {...props} />
            ) : (
              <pre className="hp-md-pre">
                <code className="hp-md-code-block" {...props} />
              </pre>
            ),
          blockquote: ({ node, ...props }) => <blockquote className="hp-md-blockquote" {...props} />,
          hr: ({ node, ...props }) => <hr className="hp-md-divider" {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
