"""Check raw base/adapted generations on held-out quotation and abstention tasks."""

import argparse
import json
from pathlib import Path
import random
import re
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from qanoon_ai.llm.client import LocalLLMClient, validate_quotation
from qanoon_ai.llm.prompts import ABSTENTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BACKEND / "training/qanoon-model")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    import torch

    client = LocalLLMClient(args.model, BACKEND.parent / "data/cache/huggingface/hub")
    if not client.ready:
        raise RuntimeError("Completed local model is not loadable")
    training = json.loads((args.model / "training-report.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (Path(training["run_dir"]) / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(2026)
    positive = [row for row in records if row["answer"] != ABSTENTION]
    negative = [row for row in records if row["answer"] == ABSTENTION]
    selected = rng.sample(positive, min(args.samples, len(positive))) + rng.sample(negative, min(5, len(negative)))
    report = {"samples": len(selected), "task": training["task"], "base": [], "adapted": []}

    def generate(messages):
        prompt = client._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = client._tokenizer(prompt, return_tensors="pt").to(client._model.device)
        with torch.inference_mode():
            result = client._model.generate(
                **inputs, max_new_tokens=192, do_sample=False,
                pad_token_id=client._tokenizer.pad_token_id,
                eos_token_id=client._tokenizer.eos_token_id,
            )
        return client._tokenizer.decode(result[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    for index, row in enumerate(selected):
        source_block = row["messages"][-1]["content"].split("\n\nSources:\n", 1)[1]
        source_text = re.sub(r"^\[1\] [^\n]*\n", "", source_block)
        for mode in ("base", "adapted"):
            if mode == "base":
                with client._model.disable_adapter():
                    text = generate(row["messages"])
            else:
                text = generate(row["messages"])
            expected_abstention = row["answer"] == ABSTENTION
            passed = text == ABSTENTION if expected_abstention else (
                text != ABSTENTION and validate_quotation(text, [source_text])
            )
            report[mode].append({"document": row["document"], "abstention_task": expected_abstention, "passed": passed, "output": text})
        print(f"Evaluated {index + 1}/{len(selected)}", flush=True)
    report["summary"] = {
        mode: {
            "passed": sum(row["passed"] for row in report[mode]),
            "quotation_passed": sum(row["passed"] for row in report[mode] if not row["abstention_task"]),
            "abstention_passed": sum(row["passed"] for row in report[mode] if row["abstention_task"]),
        } for mode in ("base", "adapted")
    }
    target = args.model / "generation-evaluation.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Report: {target}", flush=True)


if __name__ == "__main__":
    main()
