import React, { useEffect, useMemo, useState } from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";

/** Same-origin when UI is served by demo/server.py on :8000. Otherwise API on same hostname :8000 (not 127.0.0.1 vs localhost mismatch). */
function getApiBase() {
  if (typeof window === "undefined") return "";
  const { protocol, hostname, port } = window.location;
  if (protocol === "file:") return "http://127.0.0.1:8000";
  if (port === "8000") return "";
  return `${protocol}//${hostname}:8000`;
}

function DocHits({ rows }) {
  if (!rows || !rows.length) {
    return React.createElement("p", { className: "muted-small" }, "No results.");
  }
  return React.createElement(
    "div",
    { className: "doc-hits" },
    rows.map((r, i) =>
      React.createElement(
        "article",
        { key: `${r.doc_id}-${i}`, className: "doc-hit" },
        React.createElement(
          "div",
          { className: "doc-hit-meta mono" },
          `doc ${r.doc_id} · BM25 ${typeof r.score === "number" ? r.score.toFixed(3) : r.score}`
        ),
        React.createElement("p", { className: "doc-hit-body" }, r.snippet || "(no text in corpus)")
      )
    )
  );
}

function App() {
  const [query, setQuery] = useState("stock market crash 1987");
  const [feedbackText, setFeedbackText] = useState(
    "The 1987 stock market crash known as Black Monday saw Dow Jones drop 22 percent.\n" +
      "Stock market volatility increased during October 1987 financial crisis.\n" +
      "Market crash of 1987 triggered circuit breakers and trading halts.\n" +
      "Program trading blamed for 1987 crash severity."
  );
  const [examples, setExamples] = useState([]);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    const base = getApiBase();
    fetch(`${base}/examples`)
      .then((r) => r.json())
      .then((data) => setExamples(Array.isArray(data.examples) ? data.examples : []))
      .catch(() => setExamples([]));
  }, []);

  const docs = useMemo(
    () => feedbackText.split("\n").map((s) => s.trim()).filter(Boolean),
    [feedbackText]
  );

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResult(null);
      return;
    }
    setStatus("loading");
    setError("");

    const handle = setTimeout(() => {
      const base = getApiBase();
      fetch(`${base}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, feedback_docs: docs }),
      })
        .then(async (r) => {
          const text = await r.text();
          try {
            return JSON.parse(text);
          } catch {
            throw new Error("Backend not reachable on port 8000. Start demo/server.py.");
          }
        })
        .then((payload) => {
          if (!payload.ok) {
            throw new Error(payload.error || "Failed to analyze");
          }
          setResult(payload.result);
          setStatus("ok");
        })
        .catch((err) => {
          setStatus("error");
          setError(err.message || "Request failed");
        });
    }, 250);

    return () => clearTimeout(handle);
  }, [query, docs]);

  return React.createElement(
    "div",
    { className: "container" },
    React.createElement(
      "section",
      { className: "hero" },
      React.createElement("h1", null, "Adaptive Query Expansion Lab"),
      React.createElement(
        "p",
        { className: "subtitle" },
        "Live backend analysis powered by your Python IR pipeline."
      )
    ),

    React.createElement(
      "section",
      { className: "panel" },
      React.createElement(
        "div",
        { className: "grid-two" },
        React.createElement(
          "div",
          null,
          React.createElement("label", null, "Query"),
          React.createElement("input", {
            value: query,
            onChange: (e) => {
              setQuery(e.target.value);
              setFeedbackText("");
            },
            placeholder: "Enter a query...",
          }),
          React.createElement(
            "p",
            { className: "field-hint" },
            "Changing the query clears custom feedback below. Leave it empty to use top BM25 documents from the corpus for that query."
          )
        ),
        React.createElement(
          "div",
          null,
          React.createElement("label", null, "Loaded Examples"),
          React.createElement(
            "div",
            { className: "btn-row" },
            examples.map((ex) =>
              React.createElement(
                "button",
                {
                  key: ex.id,
                  onClick: () => {
                    setQuery(ex.query);
                    setFeedbackText(ex.docs.join("\n"));
                  },
                },
                ex.label
              )
            )
          )
        )
      ),
      React.createElement("label", { style: { marginTop: "12px" } }, "Feedback Documents (one per line, optional)"),
      docs.length === 0 &&
        React.createElement(
          "p",
          { className: "field-hint" },
          "Empty — backend uses the top pseudo-relevant documents from BM25 for your query."
        ),
      React.createElement("textarea", {
        value: feedbackText,
        onChange: (e) => setFeedbackText(e.target.value),
        placeholder: "Paste feedback docs, or leave empty for automatic BM25-selected passages…",
      })
    ),

    React.createElement(
      "section",
      { className: "panel" },
      React.createElement(
        "div",
        { className: "mono", style: { marginBottom: "10px" } },
        status === "loading"
          ? "Status: analyzing..."
          : status === "error"
            ? `Status: error (${error})`
            : "Status: ready"
      ),
      React.createElement(
        "div",
        { className: "metrics" },
        React.createElement(
          "div",
          { className: "metric" },
          React.createElement("div", { className: "label" }, "Coherence"),
          React.createElement("div", { className: "value" }, result ? result.coherence.toFixed(3) : "--"),
          React.createElement(
            "div",
            { className: `badge ${result && result.is_ambiguous ? "ambiguous" : "specific"}` },
            result ? result.classification : "N/A"
          )
        ),
        React.createElement(
          "div",
          { className: "metric" },
          React.createElement("div", { className: "label" }, "Quality"),
          React.createElement("div", { className: "value" }, result ? result.avg_quality.toFixed(3) : "--"),
          React.createElement(
            "div",
            { className: "mono" },
            result ? `kept ${result.kept_count}, filtered ${result.filtered_count}` : ""
          )
        ),
        React.createElement(
          "div",
          { className: "metric" },
          React.createElement("div", { className: "label" }, "Clusters"),
          React.createElement("div", { className: "value" }, result ? String(result.num_clusters) : "--"),
          React.createElement("div", { className: "mono" }, result ? `largest ${result.largest_cluster}` : "")
        ),
        React.createElement(
          "div",
          { className: "metric" },
          React.createElement("div", { className: "label" }, "Coverage"),
          React.createElement(
            "div",
            { className: "value" },
            result ? `${(result.coverage * 100).toFixed(1)}%` : "--"
          ),
          React.createElement("div", { className: "mono" }, "dominant cluster")
        )
      ),
      React.createElement(
        "div",
        { className: "expansion-block" },
        React.createElement("div", { className: "label", style: { marginBottom: "6px" } }, "Terms added to query"),
        (result?.expansion_terms || []).length > 0
          ? React.createElement(
              "div",
              { className: "chips" },
              result.expansion_terms.map((term) =>
                React.createElement("span", { key: term, className: "chip" }, term)
              )
            )
          : React.createElement(
              "p",
              { className: "muted-small" },
              "None — clustering did not add new stemmed terms (or they duplicate the query). Retrieval still runs."
            )
      )
    ),

    result &&
      React.createElement(
        "section",
        { className: "panel" },
        React.createElement("label", null, "Pseudo-relevance feedback (passages fed into expansion)"),
        React.createElement(
          "p",
          { className: "field-hint", style: { marginTop: "6px" } },
          "When feedback is empty, these are the top BM25 documents’ texts from your query."
        ),
        React.createElement(
          "ul",
          { className: "snippet-list" },
          (result.prf_input_snippets || []).map((s, i) =>
            React.createElement("li", { key: `in-${i}`, className: "mono snippet-li" }, s)
          )
        ),
        React.createElement("label", { style: { marginTop: "14px", display: "block" } }, "After quality filter"),
        React.createElement(
          "ul",
          { className: "snippet-list" },
          (result.prf_after_quality_snippets || []).map((s, i) =>
            React.createElement("li", { key: `fq-${i}`, className: "mono snippet-li" }, s)
          )
        )
      ),

    React.createElement(
      "section",
      { className: "panel" },
      React.createElement(
        "div",
        { className: "grid-two" },
        React.createElement(
          "div",
          null,
          React.createElement("label", null, "Original Query"),
          React.createElement("div", { className: "mono" }, query)
        ),
        React.createElement(
          "div",
          null,
          React.createElement("label", null, "Expanded Query"),
          React.createElement("div", { className: "mono" }, result?.expanded_query || "—")
        )
      ),
      React.createElement(
        "p",
        { className: "muted-small", style: { marginTop: "10px" } },
        (result?.expansion_terms || []).length > 0
          ? `Added stems: ${result.expansion_terms.join(", ")}.`
          : "No extra stems added — expanded query may match the original after stemming; BM25 still re-runs."
      ),
      React.createElement("div", { style: { marginTop: "12px" } },
        React.createElement("label", null, "Retrieved documents (BM25 on expanded query)"),
        React.createElement(
          "p",
          { className: "field-hint", style: { marginTop: "6px" } },
          "Reuters article text from the corpus (truncated for display)."
        ),
        React.createElement(DocHits, { rows: result?.top_results || [] })
      )
    )
  );
}

createRoot(document.getElementById("root")).render(React.createElement(App));
