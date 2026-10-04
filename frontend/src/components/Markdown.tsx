import { Fragment, ReactNode } from "react";

type Props = {
  text: string;
  onCite?: (index: number) => void;
};

/** Render inline markdown (bold, italic, code) and clickable [n] citations as React nodes. */
function renderInline(text: string, onCite?: (index: number) => void): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[\d+(?:\s*,\s*\d+)*\])/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      nodes.push(text.slice(last, match.index));
    }
    const token = match[0];
    if (token.startsWith("**")) {
      nodes.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      nodes.push(<code key={key++}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("[")) {
      const numbers = token.slice(1, -1).split(",").map((n) => Number(n.trim()));
      nodes.push(
        <span className="cite-group" key={key++}>
          {numbers.map((n) => (
            <button className="cite" key={n} type="button" onClick={() => onCite?.(n)} title={`Show source ${n}`}>
              {n}
            </button>
          ))}
        </span>
      );
    } else {
      nodes.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }
    last = match.index + token.length;
  }
  if (last < text.length) {
    nodes.push(text.slice(last));
  }
  return nodes;
}

/** A small, safe markdown renderer: headings, paragraphs, bullet and numbered lists. No raw HTML. */
export function Markdown({ text, onCite }: Props) {
  const blocks: ReactNode[] = [];
  const lines = text.split(/\r?\n/);
  let list: { ordered: boolean; items: string[] } | null = null;
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(<p key={blocks.length}>{renderInline(paragraph.join(" "), onCite)}</p>);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list) {
      const items = list.items.map((item, i) => <li key={i}>{renderInline(item, onCite)}</li>);
      blocks.push(list.ordered ? <ol key={blocks.length}>{items}</ol> : <ul key={blocks.length}>{items}</ul>);
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    const numbered = /^\d+[.)]\s+(.*)$/.exec(line);
    if (!line) {
      flushParagraph();
      flushList();
    } else if (heading) {
      flushParagraph();
      flushList();
      blocks.push(<h4 key={blocks.length}>{renderInline(heading[2], onCite)}</h4>);
    } else if (bullet || numbered) {
      flushParagraph();
      const ordered = Boolean(numbered);
      if (list && list.ordered !== ordered) {
        flushList();
      }
      list = list ?? { ordered, items: [] };
      list.items.push((bullet ?? numbered)![1]);
    } else if (/^(-{3,}|\*{3,})$/.test(line)) {
      flushParagraph();
      flushList();
      blocks.push(<hr key={blocks.length} />);
    } else {
      flushList();
      paragraph.push(line);
    }
  }
  flushParagraph();
  flushList();
  return <Fragment>{blocks}</Fragment>;
}
