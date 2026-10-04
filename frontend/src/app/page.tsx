"use client";

import Link from "next/link";
import {
  AlertTriangle,
  BookOpen,
  ExternalLink,
  FileSearch,
  FileText,
  Loader2,
  MessageSquarePlus,
  Scale,
  Send,
  Square
} from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { Markdown } from "@/components/Markdown";
import { ChatTurn, Citation, streamChat } from "@/lib/api";

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  warnings?: string[];
  searchQuery?: string;
  thinking?: boolean;
  reasoning?: string;
  status?: "streaming" | "done" | "error" | "stopped";
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
  "What is the punishment for theft in Pakistan?",
  "Cheque bounce hone par kya saza hai?",
  "Can a woman accused of a non-bailable offence get bail?",
  "طلاق کتنے دن بعد مؤثر ہوتی ہے؟"
];

/** Sources with their original [n] numbers; once an answer is finished, only the ones it cites. */
function shownSources(message: Message | undefined): { number: number; citation: Citation }[] {
  const all = (message?.citations ?? []).map((citation, index) => ({ number: index + 1, citation }));
  if (!message || message.status === "streaming") {
    return all;
  }
  const cited = new Set([...message.content.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)].flatMap((m) => m[1].split(",").map((n) => Number(n.trim()))));
  return all.filter((source) => cited.has(source.number));
}

function citationLocation(citation: Citation) {
  const parts: string[] = [];
  if (citation.section_ref) {
    parts.push(citation.section_ref);
  }
  if (citation.page_start) {
    parts.push(
      citation.page_end && citation.page_end !== citation.page_start
        ? `Pages ${citation.page_start}-${citation.page_end}`
        : `Page ${citation.page_start}`
    );
  }
  return parts.join(" / ");
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<{ messageId: number; source: number | null } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const nextId = useRef(1);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const panelMessage =
    messages.find((m) => m.id === selected?.messageId) ??
    [...messages].reverse().find((m) => m.role === "assistant" && shownSources(m).length);
  const panelSources = shownSources(panelMessage);

  function updateMessage(id: number, change: (message: Message) => Message) {
    setMessages((current) => current.map((m) => (m.id === id ? change(m) : m)));
  }

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) {
      return;
    }
    const user: Message = { id: nextId.current++, role: "user", content: question };
    const assistant: Message = { id: nextId.current++, role: "assistant", content: "", status: "streaming" };
    const history: ChatTurn[] = [...messages, user]
      .filter((m) => m.content && m.status !== "error")
      .map((m) => ({ role: m.role, content: m.content }));
    setMessages((current) => [...current, user, assistant]);
    setSelected({ messageId: assistant.id, source: null });
    setInput("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(
        history,
        jurisdiction || null,
        (event) => {
          if (event.type === "sources") {
            updateMessage(assistant.id, (m) => ({ ...m, citations: event.citations, searchQuery: event.search_query }));
          } else if (event.type === "thinking") {
            updateMessage(assistant.id, (m) => ({ ...m, thinking: true, reasoning: (m.reasoning ?? "") + event.text }));
          } else if (event.type === "token") {
            updateMessage(assistant.id, (m) => ({ ...m, content: m.content + event.text }));
          } else if (event.type === "done") {
            updateMessage(assistant.id, (m) => ({ ...m, warnings: event.warnings, status: "done" }));
          } else if (event.type === "error") {
            updateMessage(assistant.id, (m) => ({ ...m, warnings: [event.message], status: "error" }));
          }
        },
        controller.signal
      );
    } catch (err) {
      const stopped = controller.signal.aborted;
      updateMessage(assistant.id, (m) => ({
        ...m,
        status: stopped ? "stopped" : "error",
        warnings: stopped ? m.warnings : [err instanceof Error ? err.message : "Request failed"]
      }));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void send(input);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send(input);
    }
  }

  function newChat() {
    abortRef.current?.abort();
    setMessages([]);
    setSelected(null);
  }

  return (
    <main className="chat-shell">
      <aside className="sidebar" aria-label="Workspace controls">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Scale size={24} />
          </div>
          <div>
            <p className="eyebrow">Qanoon AI</p>
            <h1>Pakistani law assistant</h1>
          </div>
        </div>

        <button className="new-chat" type="button" onClick={newChat}>
          <MessageSquarePlus size={18} />
          <span>New chat</span>
        </button>

        <Link className="sidebar-link" href="/documents">
          <FileSearch size={16} />
          <span>Read a document</span>
        </Link>

        <section className="control-group" aria-labelledby="jurisdiction-label">
          <div className="control-label" id="jurisdiction-label">
            <BookOpen size={16} />
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
          Answers are based on indexed Pakistani laws and court judgments. General legal information, not legal advice.
        </p>
      </aside>

      <section className="chat-main" aria-label="Conversation">
        <div className="thread">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Ask about Pakistani law</h2>
              <p>In English, Urdu or Roman Urdu. Follow-up questions keep the context of the conversation.</p>
              <div className="example-row">
                {examples.map((example) => (
                  <button key={example} type="button" dir="auto" onClick={() => void send(example)}>
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) =>
              message.role === "user" ? (
                <div className="bubble user" key={message.id} dir="auto">
                  {message.content}
                </div>
              ) : (
                <article
                  className={`bubble assistant${selected?.messageId === message.id ? " selected" : ""}`}
                  key={message.id}
                  onClick={() => setSelected((s) => (s?.messageId === message.id ? s : { messageId: message.id, source: null }))}
                >
                  <div className="answer-body" dir="auto">
                    {message.reasoning ? (
                      <details className="reasoning-block" open={message.status === "streaming" && !message.content}>
                        <summary>
                          {message.status === "streaming" && !message.content ? (
                            <Loader2 className="spin" size={14} />
                          ) : null}
                          <span>{message.status === "streaming" && !message.content ? "Thinking…" : "Thought process"}</span>
                        </summary>
                        <p className="reasoning-text" dir="auto">{message.reasoning}</p>
                      </details>
                    ) : null}
                    {message.content ? (
                      <Markdown
                        text={message.content}
                        onCite={(n) => setSelected({ messageId: message.id, source: n })}
                      />
                    ) : message.status === "streaming" && !message.reasoning ? (
                      <span className="thinking">
                        <Loader2 className="spin" size={16} /> Searching Pakistani laws and judgments…
                      </span>
                    ) : null}
                    {message.status === "streaming" && message.content ? <span className="cursor" /> : null}
                  </div>
                  {message.status === "stopped" ? <p className="status-note">Stopped.</p> : null}
                  {message.warnings?.length ? (
                    <div className="warning-list">
                      {message.warnings.map((warning) => (
                        <div key={warning} className="warning-item">
                          <AlertTriangle size={16} />
                          <span>{warning}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {message.status !== "streaming" && shownSources(message).length ? (
                    <button
                      className="sources-chip"
                      type="button"
                      onClick={() => setSelected({ messageId: message.id, source: null })}
                    >
                      <FileText size={14} /> {shownSources(message).length} cited sources
                    </button>
                  ) : null}
                </article>
              )
            )
          )}
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={submit}>
          <textarea
            maxLength={4000}
            dir="auto"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question about Pakistani law…"
            rows={2}
          />
          {busy ? (
            <button className="send-button" type="button" onClick={() => abortRef.current?.abort()}>
              <Square size={16} />
              <span>Stop</span>
            </button>
          ) : (
            <button className="send-button" type="submit" disabled={!input.trim()}>
              <Send size={18} />
              <span>Send</span>
            </button>
          )}
        </form>
      </section>

      <aside className="citations" aria-label="Sources">
        <div className="panel-heading">
          <FileText size={18} />
          <h2>Sources</h2>
        </div>
        {panelMessage?.searchQuery ? <p className="search-query">Searched for: {panelMessage.searchQuery}</p> : null}
        <div className="citation-list">
          {panelMessage && panelSources.length ? (
            panelSources.map(({ number, citation }) => {
              const active = selected?.messageId === panelMessage.id && selected.source === number;
              return (
                <article className={`citation-card${active ? " active" : ""}`} key={citation.chunk_id ?? number}>
                  <div className="citation-title-row">
                    <h3>
                      [{number}] {citation.title}
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
                    {citationLocation(citation) ? <span>{citationLocation(citation)}</span> : null}
                  </div>
                  <details open={active}>
                    <summary>Read source passage</summary>
                    <p dir="auto">{citation.excerpt}</p>
                  </details>
                  <div className="citation-footer">
                    <span>{citation.year ?? "Unknown year"}</span>
                    <span>{citation.legal_status?.replaceAll("_", " ") ?? "Status unverified"}</span>
                  </div>
                </article>
              );
            })
          ) : (
            <div className="empty-citations">
              <FileText size={22} />
              <span>Sources for an answer appear here.</span>
            </div>
          )}
        </div>
      </aside>
    </main>
  );
}
