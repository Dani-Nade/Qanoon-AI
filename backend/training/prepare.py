"""Prepare a quotation prototype from legacy, automatically templated examples.

These are not expert-written legal answers. Train quotation and abstention;
split by document title so passages from the same law cannot cross the split.
"""

import hashlib
import json
import re
from pathlib import Path

from qanoon_ai.llm.prompts import ABSTENTION, evidence_messages


def prepare_examples(path: Path) -> tuple[list[dict], list[dict], dict]:
    train, evaluation = [], []
    seen = set()
    rejected = duplicates = 0
    documents = {"train": set(), "eval": set()}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        header, separator, body = row.get("output", "").partition("\n\n")
        if not separator or not header.startswith("According to "):
            rejected += 1
            continue
        title = " ".join(header[len("According to "):].rstrip(":").split())
        excerpt = " ".join(body.split())[:650].rsplit(" ", 1)[0]
        if len(excerpt) < 100 or sum(c.isalpha() for c in excerpt) / len(excerpt) < 0.55:
            rejected += 1
            continue
        digest = hashlib.sha256(excerpt.encode()).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        split = "eval" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"
        documents[split].add(key)
        target = evaluation if split == "eval" else train
        # Legacy questions often request facts absent from their paired passage.
        question = f"Quote a provision from {title}."
        quotation = excerpt[:350].rsplit(" ", 1)[0]
        sentences = list(re.finditer(r"[.;](?=\s|$)", quotation))
        if sentences and sentences[-1].end() >= 100:
            quotation = quotation[:sentences[-1].end()]
        target.append({
            "messages": evidence_messages(question, [(title, excerpt)]),
            "answer": f'"{quotation}" [1]', "document": key,
        })
        if int(digest[:8], 16) % 10 == 0:
            target.append({
                "messages": evidence_messages(question, []),
                "answer": ABSTENTION, "document": key,
            })
    if not train or not evaluation:
        raise ValueError("Input must produce both training and document-held-out evaluation examples")
    assert not documents["train"] & documents["eval"]
    return train, evaluation, {
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_examples": len(train), "evaluation_examples": len(evaluation),
        "training_documents": len(documents["train"]),
        "evaluation_documents": len(documents["eval"]),
        "rejected_examples": rejected, "duplicate_excerpts_removed": duplicates,
        "task": "source quotation and no-source abstention; not expert legal reasoning",
    }


def encode_example(example: dict, tokenizer, max_length: int) -> dict:
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(
        example["messages"] + [{"role": "assistant", "content": example["answer"]}],
        tokenize=True, add_generation_prompt=False,
    )
    if complete[:len(prompt)] != prompt:
        raise ValueError("Tokenizer template does not preserve the assistant prompt prefix")
    if len(complete) > max_length:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    return {
        "input_ids": complete, "attention_mask": [1] * len(complete),
        "labels": [-100] * len(prompt) + complete[len(prompt):],
    }
