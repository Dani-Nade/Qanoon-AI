"""
Generate Q&A training pairs from chunks.jsonl.
Reads  : data/chunks/chunks.jsonl
Writes : data/training/qa_pairs.jsonl

Each output record (instruction-tuning format):
{
  "instruction": "What does Pakistani law say about X?",
  "input": "",
  "output": "According to <title>: <relevant text>"
}

Templates cover: definition, punishment, rights, procedure, scope questions.
"""

import json
import random
import re
from pathlib import Path

CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks" / "chunks.jsonl"
OUT_DIR = Path(__file__).parent.parent / "data" / "training"
OUT_FILE = OUT_DIR / "qa_pairs.jsonl"

# Instruction templates — varied so model learns different query styles
_TEMPLATES = [
    "What does Pakistani law say about {topic}?",
    "Explain the legal provisions regarding {topic} under Pakistani law.",
    "What are the rules for {topic} according to Pakistani statutes?",
    "Describe the punishment or consequences for {topic} in Pakistan.",
    "What rights does a person have regarding {topic} under Pakistani law?",
    "How does Pakistani law define {topic}?",
    "What is the legal procedure for {topic} in Pakistan?",
    "Summarize the key provisions of {topic} under Pakistani legislation.",
    "Under which law is {topic} regulated in Pakistan?",
    "What are the legal requirements for {topic} in Pakistan?",
]


def _extract_topic(text: str, title: str) -> str:
    """Extract a meaningful topic from chunk text or fall back to title."""
    # Try to grab first noun phrase after a section marker
    match = re.search(
        r"(?:means?|refers? to|defined as|includes?)\s+([^\.]{10,60})", text, re.I
    )
    if match:
        return match.group(1).strip().lower()

    # Fall back: use the law title, cleaned up
    topic = re.sub(r"\b(Act|Ordinance|Order|Rules?|Regulations?),?\s*\d{4}", "", title)
    topic = re.sub(r"^\w+\s+of\s+\d{4}\s*-\s*", "", topic)
    return topic.strip().lower() or title.lower()


def _make_pair(chunk: dict) -> dict:
    topic = _extract_topic(chunk["text"], chunk["title"])
    instruction = random.choice(_TEMPLATES).format(topic=topic)
    output = f"According to {chunk['title']} ({chunk['year']}):\n\n{chunk['text']}"
    return {"instruction": instruction, "input": "", "output": output}


def run(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists() and not force:
        print(f"QA pairs already exist at {OUT_FILE}")
        return

    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Generating Q&A pairs from {len(chunks)} chunks...")

    random.seed(42)
    pairs = [_make_pair(c) for c in chunks]

    # shuffle so training sees varied years/topics
    random.shuffle(pairs)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Generated : {len(pairs)} Q&A pairs")
    print(f"Output    : {OUT_FILE}")


if __name__ == "__main__":
    run()
