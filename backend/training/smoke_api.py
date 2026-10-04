"""Exercise the running API and record quotation/fallback behavior."""

import json
from pathlib import Path
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import requests
from qanoon_ai.llm.client import validate_quotation


def main():
    base = "http://127.0.0.1:8000"
    health = requests.get(f"{base}/health", timeout=60)
    health.raise_for_status()
    report = {"health": health.json(), "queries": []}
    if not (report["health"].get("llm_mode") == "quotation"
            and report["health"].get("model_ready")
            and report["health"].get("trained_adapter_available")):
        raise RuntimeError("Start the API with QANOON_LLM_MODE=quotation and a completed, loadable adapter")
    for question in (
        "Quote a provision from the Civil Servants Act.",
        "What is the punishment for theft in Pakistan?",
        "Pakistan mein chori ki saza kya hai?",
        "How do I renew a driving licence on Mars?",
    ):
        start = time.monotonic()
        response = requests.post(f"{base}/query", json={"question": question, "max_sources": 3}, timeout=90)
        response.raise_for_status()
        answer = response.json()
        answer["latency_seconds"] = round(time.monotonic() - start, 2)
        answer["validated_model_format"] = validate_quotation(
            answer["answer"], [citation["excerpt"] for citation in answer["citations"]],
        )
        report["queries"].append(answer)
        print(json.dumps({
            "question": question, "model_ready": answer["model_ready"],
            "citations": len(answer["citations"]), "warnings": answer["warnings"],
            "validated_model_format": answer["validated_model_format"],
            "latency_seconds": answer["latency_seconds"],
        }), flush=True)
    target = BACKEND / "training/qanoon-model/api-evaluation.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {target}", flush=True)


if __name__ == "__main__":
    main()
