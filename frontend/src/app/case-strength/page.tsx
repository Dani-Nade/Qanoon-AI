"use client";

import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  ExternalLink,
  FileText,
  Gavel,
  Loader2,
  Square
} from "lucide-react";
import { FormEvent, useRef, useState } from "react";
import { Markdown } from "@/components/Markdown";
import { Citation, streamCaseStrength } from "@/lib/api";

type Result = {
  content: string;
  citations: Citation[];
  warnings: string[];
  searchQuery?: string;
  thinking: boolean;
  reasoning: string;
  status: "streaming" | "done" | "error" | "stopped";
};

const jurisdictions = [
  { label: "All", value: "" },
  { label: "Federal", value: "federal" },
  { label: "Punjab", value: "punjab" },
  { label: "Sindh", value: "sindh" },
  { label: "KP", value: "kp" },
  { label: "Balochistan", value: "balochistan" },
  { label: "ICT", value: "ict" }
];

const examples = [
  "Landlord is evicting me without any written notice, I've paid rent every month on time.",
  "Meri biwi ne jhootay ilzaam laga kar khula mang liya hai, mujhe koi notice nahi mila.",
  "I was fired from my job two days after reporting a safety issue to my manager."
];

const strengthTone: Record<string, string> = {
  high: "good",
  medium: "warn",
  low: "bad"
};

function detectStrength(text: string): { label: string; tone: string } | null {
  const match = text.match(/Case strength:\s*(Low|Medium|High|Not enough information)/i);
  if (!match) return null;
  const label = match[1];
  const tone = strengthTone[label.toLowerCase()] ?? "muted";
  return { label, tone };
}

export default function CaseStrengthPage() {
  const [facts, setFacts] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = facts.trim();
    if (!text || busy) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setResult({ content: "", citations: [], warnings: [], thinking: false, reasoning: "", status: "streaming" });

    try {
      await streamCaseStrength(
        text,
        jurisdiction || null,
        (event) => {
          setResult((current) => {
            if (!current) return current;
            if (event.type === "sources") return { ...current, citations: event.citations, searchQuery: event.search_query };
            if (event.type === "thinking") return { ...current, thinking: true, reasoning: current.reasoning + event.text };
            if (event.type === "token") return { ...current, content: current.content + event.text };
            if (event.type === "done") return { ...current, warnings: event.warnings, status: "done" };
            return { ...current, warnings: [event.message], status: "error" };
          });
        },
        controller.signal
      );
    } catch (err) {
      setResult((current) => current && {
        ...current,
        status: controller.signal.aborted ? "stopped" : "error",
        warnings: controller.signal.aborted ? current.warnings : [err instanceof Error ? err.message : "Request failed"]
      });
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  const strength = result?.status !== "streaming" || result.content ? detectStrength(result?.content ?? "") : null;
  const bodyText = result ? result.content.replace(/^Case strength:.*\n*/i, "").trim() : "";

  return (
    <main className="chat-shell">
      <aside className="sidebar" aria-label="Workspace controls">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Gavel size={24} />
          </div>
          <div>
            <p className="eyebrow">Qanoon AI</p>
            <h1>Case strength</h1>
          </div>
        </div>

        <Link className="sidebar-link" href="/">
          <ArrowLeft size={16} />
          <span>Back to chat</span>
        </Link>

        <section className="control-group" aria-labelledby="jurisdiction-label">
          <div className="control-label" id="jurisdiction-label">
            <span>Jurisdiction</span>
          </div>
          <div className="jurisdiction-grid">
            {jurisdictions.map((option) => (
              <button
                key={option.label}
                className={jurisdiction === option.value ? "active" : ""}
                type="button"
                onClick={() => setJurisdiction(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </section>

        <p className="sidebar-note">
          Compares your facts against similar Pakistani court judgments already indexed. This is a general
          comparison with past cases, not a prediction of your result or legal advice.
        </p>
      </aside>

      <section className="chat-main" aria-label="Case strength estimator">
        <div className="thread">
          {!result ? (
            <div className="empty-state">
              <h2>Describe your situation</h2>
              <p>In English, Urdu or Roman Urdu. Include what happened, when, and anything you can show as evidence.</p>
              <div className="example-row">
                {examples.map((example) => (
                  <button key={example} type="button" dir="auto" onClick={() => setFacts(example)}>
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <article className="bubble assistant selected">
              {strength ? (
                <div className={`sources-chip strength-${strength.tone}`}>
                  <Gavel size={14} /> {strength.label}
                </div>
              ) : null}
              <div className="answer-body" dir="auto">
                {result.reasoning ? (
                  <details className="reasoning-block" open={result.status === "streaming" && !result.content}>
                    <summary>
                      {result.status === "streaming" && !result.content ? <Loader2 className="spin" size={14} /> : null}
                      <span>{result.status === "streaming" && !result.content ? "Thinking…" : "Thought process"}</span>
                    </summary>
                    <p className="reasoning-text" dir="auto">{result.reasoning}</p>
                  </details>
                ) : null}
                {bodyText ? (
                  <Markdown text={bodyText} onCite={() => undefined} />
                ) : result.status === "streaming" && !result.reasoning ? (
                  <span className="thinking">
                    <Loader2 className="spin" size={16} /> Searching similar judgments…
                  </span>
                ) : null}
                {result.status === "streaming" && result.content ? <span className="cursor" /> : null}
              </div>
              {result.status === "stopped" ? <p className="status-note">Stopped.</p> : null}
              {result.warnings.length ? (
                <div className="warning-list">
                  {result.warnings.map((warning) => (
                    <div key={warning} className="warning-item">
                      <AlertTriangle size={16} />
                      <span>{warning}</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </article>
          )}
        </div>

        <form className="composer" onSubmit={submit}>
          <textarea
            maxLength={8000}
            dir="auto"
            value={facts}
            onChange={(event) => setFacts(event.target.value)}
            placeholder="Describe the facts of your case…"
            rows={2}
          />
          {busy ? (
            <button className="send-button" type="button" onClick={() => abortRef.current?.abort()}>
              <Square size={16} />
              <span>Stop</span>
            </button>
          ) : (
            <button className="send-button" type="submit" disabled={facts.trim().length < 15}>
              <Gavel size={18} />
              <span>Estimate</span>
            </button>
          )}
        </form>
      </section>

      <aside className="citations" aria-label="Precedents">
        <div className="panel-heading">
          <FileText size={18} />
          <h2>Precedents</h2>
        </div>
        {result?.searchQuery ? <p className="search-query">Searched for: {result.searchQuery}</p> : null}
        <div className="citation-list">
          {result && result.citations.length ? (
            result.citations.map((citation, index) => (
              <article className="citation-card" key={citation.chunk_id ?? index}>
                <div className="citation-title-row">
                  <h3>
                    [{index + 1}] {citation.title}
                  </h3>
                  {citation.source_url ? (
                    <a
                      aria-label={`Open official source for ${citation.title}`}
                      href={citation.source_url}
                      rel="noreferrer"
                      target="_blank"
                      title="Open official source"
                    >
                      <ExternalLink size={15} />
                    </a>
                  ) : null}
                </div>
                <div className="citation-meta">
                  {citation.jurisdiction ? <span>{citation.jurisdiction.replaceAll("_", " ")}</span> : null}
                  {citation.document_type ? <span>{citation.document_type}</span> : null}
                </div>
                <details>
                  <summary>Read source passage</summary>
                  <p dir="auto">{citation.excerpt}</p>
                </details>
                <div className="citation-footer">
                  <span>{citation.year ?? "Unknown year"}</span>
                  <span>{citation.legal_status?.replaceAll("_", " ") ?? "Status unverified"}</span>
                </div>
              </article>
            ))
          ) : (
            <div className="empty-citations">
              <FileText size={22} />
              <span>Similar judgments will appear here.</span>
            </div>
          )}
        </div>
      </aside>
    </main>
  );
}
