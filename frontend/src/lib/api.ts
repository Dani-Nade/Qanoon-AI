export type Citation = {
  title: string;
  source_file: string;
  source_id?: string | null;
  source_url?: string | null;
  chunk_id?: string | null;
  year?: string | null;
  jurisdiction?: string | null;
  document_type?: string | null;
  legal_status?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  section_ref?: string | null;
  score?: number | null;
  excerpt: string;
};

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

export type ChatEvent =
  | { type: "sources"; citations: Citation[]; language: string; search_query: string }
  | { type: "thinking"; text: string }
  | { type: "token"; text: string }
  | { type: "done"; warnings: string[] }
  | { type: "error"; message: string };

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export type DocumentBlock = {
  index: number;
  label: string;
  bbox: [number, number, number, number];
  confidence: number | null;
  text: string;
  original_text: string;
  handwritten: boolean;
  uncertain: string[];
  edited: boolean;
};

export type DocumentPage = {
  number: number;
  width: number;
  height: number;
  method: "ocr" | "text-layer";
  document_type: string | null;
  language: "english" | "urdu" | "mixed" | null;
  handwriting_present: boolean;
  handwritten_items: { text: string; location: string }[];
  blocks: DocumentBlock[];
  warnings: string[];
};

export type UploadedDocument = {
  id: string;
  filename: string;
  status: "processing" | "ready" | "error";
  page_count: number;
  pages_done: number;
  error: string | null;
  created_at: string;
  expires_at: string;
  pages?: DocumentPage[];
};

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : `Backend returned ${response.status}`;
  } catch {
    return `Backend returned ${response.status}`;
  }
}

export async function uploadDocument(file: File): Promise<UploadedDocument> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE_URL}/documents`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getDocument(id: string): Promise<UploadedDocument> {
  const response = await fetch(`${API_BASE_URL}/documents/${id}`);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export function pageImageUrl(id: string, page: number): string {
  return `${API_BASE_URL}/documents/${id}/pages/${page}/image`;
}

export async function saveCorrections(id: string, page: number, blocks: { index: number; text: string }[]): Promise<DocumentPage> {
  const response = await fetch(`${API_BASE_URL}/documents/${id}/pages/${page}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blocks })
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/documents/${id}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) throw new Error(await readError(response));
}

/** Match the backend's eight history turns plus the latest question. */
export function prepareChatHistory(messages: ChatTurn[]): ChatTurn[] {
  return messages.slice(-9).map((message, index, recent) => {
    // Bound earlier replies (which may be longer than an API request allows),
    // while keeping the user's latest question intact for validation.
    if (index === recent.length - 1) return message;
    const characters = Array.from(message.content);
    const suffix = "\n[Earlier message shortened]";
    return {
      ...message,
      content: characters.length <= 8000 ? message.content
        : characters.slice(0, 8000 - suffix.length).join("") + suffix
    };
  });
}

/** Stream a chat reply; calls onEvent for each newline-delimited JSON event. */
export async function streamChat(
  messages: ChatTurn[],
  jurisdiction: string | null,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: prepareChatHistory(messages), jurisdiction }),
    signal
  });
  if (!response.ok || !response.body) {
    throw new Error(`Backend returned ${response.status}`);
  }
  await readEvents(response, onEvent);
}

/** Stream a case strength estimate; the events match /chat. */
export async function streamCaseStrength(
  facts: string,
  jurisdiction: string | null,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/case-strength`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ facts, jurisdiction }),
    signal
  });
  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }
  await readEvents(response, onEvent);
}

/** Stream a document explanation; the events match /chat. */
export async function streamDocumentExplanation(
  id: string,
  language: "english" | "urdu" | "roman_urdu",
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/documents/${id}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language }),
    signal
  });
  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }
  await readEvents(response, onEvent);
}

async function readEvents(response: Response, onEvent: (event: ChatEvent) => void): Promise<void> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) {
        onEvent(JSON.parse(line) as ChatEvent);
      }
      newline = buffer.indexOf("\n");
    }
  }
  if (buffer.trim()) {
    onEvent(JSON.parse(buffer) as ChatEvent);
  }
}
