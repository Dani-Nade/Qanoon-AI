"use client";

import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  FileSearch,
  Loader2,
  PenLine,
  Save,
  Sparkles,
  Square,
  Trash2,
  Upload
} from "lucide-react";
import { ChangeEvent, DragEvent, Fragment, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Markdown } from "@/components/Markdown";
import {
  Citation,
  deleteDocument,
  DocumentBlock,
  DocumentPage,
  getDocument,
  pageImageUrl,
  saveCorrections,
  streamDocumentExplanation,
  UploadedDocument,
  uploadDocument
} from "@/lib/api";

type Explanation = {
  text: string;
  citations: Citation[];
  warnings: string[];
  status: "streaming" | "done" | "error" | "stopped";
  thinking: boolean;
  reasoning: string;
};

const explainLanguages = [
  { label: "English", value: "english" },
  { label: "اردو", value: "urdu" },
  { label: "Roman Urdu", value: "roman_urdu" }
] as const;

const ACCEPT = ".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp";

/** Show flagged strings inside a block's text as highlighted marks. */
function highlight(text: string, flags: string[]): ReactNode {
  const parts = flags.filter(Boolean).sort((a, b) => b.length - a.length);
  if (!parts.length) return text;
  const escaped = parts.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pieces = text.split(new RegExp(`(${escaped.join("|")})`, "g"));
  return pieces.map((piece, i) => (parts.includes(piece) ? <mark key={i}>{piece}</mark> : <Fragment key={i}>{piece}</Fragment>));
}

function blockState(block: DocumentBlock): "handwritten" | "uncertain" | "edited" | "plain" {
  if (block.handwritten) return "handwritten";
  if (block.uncertain.length) return "uncertain";
  if (block.edited) return "edited";
  return "plain";
}

export default function DocumentsPage() {
  const [document, setDocument] = useState<UploadedDocument | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [selected, setSelected] = useState<number | null>(null);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [explainLanguage, setExplainLanguage] = useState<(typeof explainLanguages)[number]["value"]>("english");
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const explainAbort = useRef<AbortController | null>(null);

  async function explain() {
    if (!document) return;
    explainAbort.current?.abort();
    const controller = new AbortController();
    explainAbort.current = controller;
    setExplanation({ text: "", citations: [], warnings: [], status: "streaming", thinking: false, reasoning: "" });
    try {
      await streamDocumentExplanation(document.id, explainLanguage, (event) => {
        setExplanation((current) => {
          if (!current) return current;
          if (event.type === "sources") return { ...current, citations: event.citations };
          if (event.type === "thinking") return { ...current, thinking: true, reasoning: current.reasoning + event.text };
          if (event.type === "token") return { ...current, text: current.text + event.text };
          if (event.type === "done") return { ...current, warnings: event.warnings, status: "done" };
          return { ...current, warnings: [event.message], status: "error" };
        });
      }, controller.signal);
    } catch (err) {
      setExplanation((current) => current && {
        ...current,
        status: controller.signal.aborted ? "stopped" : "error",
        warnings: controller.signal.aborted ? current.warnings : [err instanceof Error ? err.message : "Explanation failed"]
      });
    }
  }

  const page: DocumentPage | undefined = useMemo(
    () => document?.pages?.find((p) => p.number === pageNumber),
    [document, pageNumber]
  );

  // Poll while pages are being read.
  useEffect(() => {
    if (!document || document.status !== "processing") return;
    const timer = setInterval(async () => {
      try {
        setDocument(await getDocument(document.id));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not load the document");
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [document]);

  useEffect(() => {
    setSelected(null);
    setDrafts({});
  }, [pageNumber, document?.id]);

  const start = useCallback(async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    setDocument(null);
    setPageNumber(1);
    explainAbort.current?.abort();
    setExplanation(null);
    try {
      const uploaded = await uploadDocument(file);
      setDocument(await getDocument(uploaded.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }, []);

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    void start(event.dataTransfer.files[0]);
  }

  async function save() {
    if (!document || !page) return;
    const changes = Object.entries(drafts).map(([index, text]) => ({ index: Number(index), text }));
    if (!changes.length) return;
    try {
      const updated = await saveCorrections(document.id, page.number, changes);
      setDocument({ ...document, pages: document.pages?.map((p) => (p.number === updated.number ? updated : p)) });
      setDrafts({});
      setNotice(`Saved ${changes.length} correction${changes.length > 1 ? "s" : ""} on page ${page.number}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Saving failed");
    }
  }

  async function remove() {
    if (!document) return;
    try {
      await deleteDocument(document.id);
      setDocument(null);
      setNotice("Deleted the document and everything read from it.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  const flaggedOnPage = page?.blocks.filter((b) => b.handwritten || b.uncertain.length).length ?? 0;

  return (
    <main className="doc-shell">
      <header className="doc-header">
        <Link className="back-link" href="/">
          <ArrowLeft size={16} /> Back to chat
        </Link>
        <div>
          <p className="eyebrow">Qanoon AI · document reader (test)</p>
          <h1>Check how a legal document is read</h1>
          <p className="doc-intro">
            Upload a scanned or photographed legal document in Urdu or English. Each page is read by Surya OCR and
            checked by the vision model for handwriting. Compare the text with the page, correct mistakes, and see what
            was flagged.
          </p>
        </div>
      </header>

      <label
        className={`drop-zone${dragging ? " dragging" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          id="document-file"
          type="file"
          accept={ACCEPT}
          onChange={(e: ChangeEvent<HTMLInputElement>) => void start(e.target.files?.[0])}
          disabled={busy}
        />
        {busy ? <Loader2 className="spin" size={22} /> : <Upload size={22} />}
        <span>
          <strong>{busy ? "Uploading…" : "Choose a file or drop it here"}</strong>
          PDF or image (PNG, JPG, WEBP, TIFF), up to 20 pages and 25 MB. Deleted automatically after 24 hours.
        </span>
      </label>

      {error ? (
        <div className="alert error" role="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      ) : null}
      {notice ? <p className="doc-notice">{notice}</p> : null}

      {document ? (
        <section className="doc-result" aria-label="Reading result">
          <div className="doc-status-row">
            <div>
              <strong>{document.filename}</strong>
              <span className="muted">
                {document.status === "processing"
                  ? ` · Reading page ${Math.min(document.pages_done + 1, document.page_count)} of ${document.page_count}…`
                  : document.status === "ready"
                    ? ` · ${document.page_count} page${document.page_count > 1 ? "s" : ""} read`
                    : " · Reading failed"}
              </span>
            </div>
            <button className="ghost-button" type="button" onClick={remove}>
              <Trash2 size={16} /> Delete now
            </button>
          </div>
          {document.status === "processing" ? (
            <div className="progress" aria-hidden="true">
              <span style={{ width: `${(document.pages_done / Math.max(document.page_count, 1)) * 100}%` }} />
            </div>
          ) : null}
          {document.status === "error" ? (
            <div className="alert error">
              <AlertTriangle size={18} />
              <span>{document.error}</span>
            </div>
          ) : null}

          {document.status === "ready" ? (
            <section className="explain-panel" aria-label="Explanation">
              <div className="explain-head">
                <div>
                  <h2>What this document is about</h2>
                  <p className="muted">
                    Explained from the text above (including your corrections) and the laws it cites. Handwritten or
                    uncertain values are listed separately, never stated as fact.
                  </p>
                </div>
                <div className="explain-controls">
                  <div className="segmented" role="group" aria-label="Explanation language">
                    {explainLanguages.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        className={explainLanguage === option.value ? "active" : ""}
                        onClick={() => setExplainLanguage(option.value)}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  {explanation?.status === "streaming" ? (
                    <button className="ghost-button" type="button" onClick={() => explainAbort.current?.abort()}>
                      <Square size={14} /> Stop
                    </button>
                  ) : (
                    <button className="send-button" type="button" onClick={explain}>
                      <Sparkles size={16} />
                      <span>{explanation ? "Explain again" : "Explain this document"}</span>
                    </button>
                  )}
                </div>
              </div>

              {explanation ? (
                <div className="explain-body">
                  <div className="answer-body" dir="auto">
                    {explanation.reasoning ? (
                      <details className="reasoning-block" open={explanation.status === "streaming" && !explanation.text}>
                        <summary>
                          {explanation.status === "streaming" && !explanation.text ? (
                            <Loader2 className="spin" size={14} />
                          ) : null}
                          <span>{explanation.status === "streaming" && !explanation.text ? "Thinking…" : "Thought process"}</span>
                        </summary>
                        <p className="reasoning-text" dir="auto">{explanation.reasoning}</p>
                      </details>
                    ) : null}
                    {explanation.text ? (
                      <Markdown
                        text={explanation.text}
                        onCite={(n) => globalThis.document.getElementById(`law-source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" })}
                      />
                    ) : explanation.status === "streaming" && !explanation.reasoning ? (
                      <span className="thinking">
                        <Loader2 className="spin" size={16} /> Finding the laws this document relies on…
                      </span>
                    ) : null}
                    {explanation.status === "streaming" && explanation.text ? <span className="cursor" /> : null}
                  </div>
                  {explanation.status === "stopped" ? <p className="status-note">Stopped.</p> : null}
                  {explanation.warnings.length ? (
                    <div className="warning-list">
                      {explanation.warnings.map((warning) => (
                        <div key={warning} className="warning-item">
                          <AlertTriangle size={16} />
                          <span>{warning}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {explanation.citations.length ? (
                    <details className="law-sources" open>
                      <summary>Laws used ({explanation.citations.length})</summary>
                      <ol>
                        {explanation.citations.map((citation, index) => (
                          <li key={citation.chunk_id ?? index} id={`law-source-${index + 1}`}>
                            <strong>[{index + 1}] {citation.title}</strong>
                            {citation.section_ref ? <span className="muted"> · {citation.section_ref}</span> : null}
                            <details>
                              <summary>Read the text</summary>
                              <p dir="auto">{citation.excerpt}</p>
                            </details>
                          </li>
                        ))}
                      </ol>
                    </details>
                  ) : null}
                </div>
              ) : null}
            </section>
          ) : null}

          {document.pages?.length ? (
            <>
              <nav className="page-tabs" aria-label="Pages">
                {document.pages.map((p) => {
                  const flags = p.blocks.filter((b) => b.handwritten || b.uncertain.length).length;
                  return (
                    <button
                      key={p.number}
                      type="button"
                      className={p.number === pageNumber ? "active" : ""}
                      onClick={() => setPageNumber(p.number)}
                    >
                      Page {p.number}
                      {p.handwriting_present ? <span className="chip hw">handwriting</span> : null}
                      {flags ? <span className="chip warn">{flags} to check</span> : null}
                    </button>
                  );
                })}
              </nav>

              {page ? (
                <div className="review-grid">
                  <div className="page-viewer">
                    <div className="page-frame">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={pageImageUrl(document.id, page.number)} alt={`Page ${page.number} of ${document.filename}`} />
                      {page.blocks.map((block) => {
                        const [x0, y0, x1, y1] = block.bbox;
                        return (
                          <button
                            key={block.index}
                            type="button"
                            aria-label={`Block ${block.index + 1}: ${block.label}`}
                            className={`ocr-box ${blockState(block)}${selected === block.index ? " selected" : ""}`}
                            style={{
                              left: `${(x0 / page.width) * 100}%`,
                              top: `${(y0 / page.height) * 100}%`,
                              width: `${((x1 - x0) / page.width) * 100}%`,
                              height: `${((y1 - y0) / page.height) * 100}%`
                            }}
                            onClick={() => {
                              setSelected(block.index);
                              globalThis.document.getElementById(`block-${block.index}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
                            }}
                          />
                        );
                      })}
                    </div>
                    <div className="legend">
                      <span><i className="swatch plain" /> Read</span>
                      <span><i className="swatch handwritten" /> Handwritten</span>
                      <span><i className="swatch uncertain" /> Number to check</span>
                      <span><i className="swatch edited" /> Corrected</span>
                    </div>
                  </div>

                  <div className="page-text">
                    <dl className="page-facts">
                      <div><dt>Document type</dt><dd>{page.document_type ?? "Not identified"}</dd></div>
                      <div><dt>Language</dt><dd>{page.language ?? "Unknown"}</dd></div>
                      <div><dt>Read from</dt><dd>{page.method === "ocr" ? "Scanned image (Surya OCR)" : "PDF text layer"}</dd></div>
                      <div><dt>Handwriting</dt><dd>{page.handwriting_present ? "Detected" : "None detected"}</dd></div>
                    </dl>

                    {page.handwritten_items.length ? (
                      <div className="hw-list">
                        <p className="label">Handwritten items found by the vision check</p>
                        <ul>
                          {page.handwritten_items.map((item, i) => (
                            <li key={i}>
                              <PenLine size={14} /> {item.text ? <span dir="auto">“{item.text}”</span> : "Unreadable"}
                              {item.location ? <span className="muted"> · {item.location}</span> : null}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}

                    {page.warnings.length ? (
                      <div className="warning-list">
                        {page.warnings.map((warning) => (
                          <div key={warning} className="warning-item">
                            <AlertTriangle size={16} />
                            <span>{warning}</span>
                          </div>
                        ))}
                      </div>
                    ) : null}

                    <div className="blocks-head">
                      <p className="label">
                        {page.blocks.length} text blocks{flaggedOnPage ? ` · ${flaggedOnPage} flagged` : ""}
                      </p>
                      <button className="send-button" type="button" disabled={!Object.keys(drafts).length} onClick={save}>
                        <Save size={16} />
                        <span>Save corrections</span>
                      </button>
                    </div>

                    <ol className="block-list">
                      {page.blocks.map((block) => {
                        const draft = drafts[block.index];
                        return (
                          <li
                            key={block.index}
                            id={`block-${block.index}`}
                            className={`block-card ${blockState(block)}${selected === block.index ? " selected" : ""}`}
                            onClick={() => setSelected(block.index)}
                          >
                            <div className="block-meta">
                              <span>#{block.index + 1} · {block.label}</span>
                              {block.handwritten ? <span className="chip hw">handwritten</span> : null}
                              {block.uncertain.length ? <span className="chip warn">check: {block.uncertain.join(", ")}</span> : null}
                              {block.edited ? <span className="chip ok">corrected</span> : null}
                            </div>
                            {block.uncertain.length ? (
                              <p className="ocr-text preview" dir="auto">{highlight(block.text, block.uncertain)}</p>
                            ) : null}
                            <textarea
                              id={`block-text-${page.number}-${block.index}`}
                              className="ocr-text"
                              dir="auto"
                              rows={Math.min(12, Math.max(2, Math.ceil((draft ?? block.text).length / 70)))}
                              value={draft ?? block.text}
                              onChange={(e) => setDrafts((d) => ({ ...d, [block.index]: e.target.value }))}
                            />
                            {block.edited ? (
                              <details>
                                <summary>Originally read as</summary>
                                <p className="ocr-text" dir="auto">{block.original_text}</p>
                              </details>
                            ) : null}
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                </div>
              ) : null}
            </>
          ) : document.status === "processing" ? (
            <div className="empty-citations">
              <FileSearch size={22} />
              <span>The first page appears here as soon as it has been read.</span>
            </div>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
