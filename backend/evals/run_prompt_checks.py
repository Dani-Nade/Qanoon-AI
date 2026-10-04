"""Run the eval questions against the live API and grade them against recorded expectations.

Each case in prompt_cases.json may declare:
  expect            "answer" (a verified generated answer), "decline", "clarify" (no citations, asks
                    the user a question), or "either"
  sections          section/article numbers, at least one of which must be cited
  must_include      groups of alternatives; every group needs one alternative in the answer
  must_not_include  phrases that must not appear in the answer
  jurisdiction      optional jurisdiction filter sent with the question

Expectations were checked against the indexed legal text. Passing is a regression
signal for grounding and key facts, not a legal-accuracy certification.

Usage: python evals/run_prompt_checks.py [--only id ...] [--category name ...] [--out path]
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import requests

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from qanoon_ai.verification.citations import citation_numbers
ANSWERED = {"grounded_answer", "quotation"}
URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().translate(URDU_DIGITS)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(r"[-‐‑‒–—―]", " ", text)
    return " ".join(text.split())


def section_key(text):
    return re.sub(r"^(section|article)", "", re.sub(r"[^0-9a-z]", "", (text or "").casefold()))


def cites_expected(case, section_ref, title):
    """A citation matches when both its section number and (if given) its law title match."""
    if section_key(section_ref) not in {section_key(s) for s in case.get("sections") or []}:
        return False
    title = re.sub(r"[-_\s]+", " ", (title or "").casefold())
    return not case.get("law") or any(law in title for law in case["law"])


NOT_COVERED_PREFIXES = (
    "i couldn't find this in the pakistani law documents",
    "mujhe yeh baat mere paas maujood pakistani qanooni dastavezat mein nahi mili",
    "مجھے یہ بات میرے پاس موجود پاکستانی قانونی دستاویزات میں نہیں ملی",
)


def ask_chat(base_url, case, timeout):
    """Call the streaming /chat endpoint and shape the reply like a /query result for grading."""
    response = requests.post(f"{base_url}/chat", timeout=timeout, stream=True, json={
        "messages": [{"role": "user", "content": case["question"]}], "jurisdiction": case.get("jurisdiction"),
    })
    response.raise_for_status()
    sources, answer, warnings, error = [], [], [], None
    for line in response.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event["type"] == "sources":
            sources = event["citations"]
        elif event["type"] == "token":
            answer.append(event["text"])
        elif event["type"] == "done":
            warnings = event["warnings"]
        elif event["type"] == "error":
            error = event["message"]
    text = "".join(answer)
    declined = normalize(text).startswith(tuple(normalize(p) for p in NOT_COVERED_PREFIXES)) or not text.strip()
    # Only sources the answer actually cites count, so an off-topic answer cannot pass on retrieval alone.
    cited = set(citation_numbers(text))
    return {
        "answer": text, "answer_mode": "error" if error else ("insufficient_evidence" if declined else "grounded_answer"),
        "citations": [c for i, c in enumerate(sources, 1) if i in cited], "retrieved": sources,
        "warnings": warnings + ([error] if error else []),
    }


def grade(case, result):
    failures = []
    mode = result["answer_mode"]
    expect = case.get("expect", "either")
    if expect == "answer" and mode not in ANSWERED:
        failures.append(f"expected an answer, got {mode}")
    if expect == "decline" and mode != "insufficient_evidence":
        failures.append(f"expected a decline, got {mode}")
    if expect == "clarify":
        # Vague messages and greetings: no legal answer or citations, and a question back to the user.
        if result["citations"]:
            failures.append("expected no citations for a vague message or greeting")
        if "?" not in result["answer"] and "؟" not in result["answer"]:
            failures.append("expected a question back to the user")
        return failures

    if mode in ANSWERED:
        wanted = case.get("sections") or []
        if wanted and not any(cites_expected(case, c.get("section_ref"), c.get("title")) for c in result["citations"]):
            cited = sorted({section_key(c.get("section_ref")) for c in result["citations"]} - {""})
            failures.append(f"none of sections {wanted} of {case.get('law') or 'any law'} cited (cited: {cited})")
        answer = normalize(result["answer"])
        for group in case.get("must_include") or []:
            if not any(normalize(option) in answer for option in group):
                failures.append(f"missing one of {group}")

    answer = normalize(result["answer"])
    for phrase in case.get("must_not_include") or []:
        if normalize(phrase) in answer:
            failures.append(f"contains forbidden {phrase!r}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--category", nargs="*", help="categories to run")
    parser.add_argument("--out", default=str(BACKEND.parent / "data/reports/prompt-checks-latest.json"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--endpoint", choices=["query", "chat"], default="chat")
    args = parser.parse_args()

    cases = json.loads((Path(__file__).parent / "prompt_cases.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [case for case in cases if case["id"] in args.only]
    if args.category:
        cases = [case for case in cases if case["category"] in args.category]

    health = requests.get(f"{args.base_url}/health", timeout=300)
    health.raise_for_status()
    report = {
        "tested_at": datetime.now(timezone.utc).isoformat(), "health": health.json(),
        "scope": "Grounding, citations and key facts checked against indexed sources; not a legal-accuracy certification",
        "results": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for case in cases:
        started = time.monotonic()
        if args.endpoint == "chat":
            result = ask_chat(args.base_url, case, args.timeout)
        else:
            response = requests.post(f"{args.base_url}/query", timeout=args.timeout, json={
                "question": case["question"], "language": "auto", "jurisdiction": case.get("jurisdiction"),
                "max_sources": 5, "require_citations": True,
            })
            response.raise_for_status()
            result = response.json()
        failures = grade(case, result)
        if result["answer_mode"] == "error":
            failures.append("chat error: " + "; ".join(result["warnings"]))
        entry = {
            **case, "latency_seconds": round(time.monotonic() - started, 2), "mode": result["answer_mode"],
            "passed": not failures, "failures": failures, "response": result,
        }
        report["results"].append(entry)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "PASS" if not failures else "FAIL"
        print(f"{status} {case['id']:<38} {result['answer_mode']:<22} {entry['latency_seconds']:>6}s  {'; '.join(failures)}", flush=True)

    by_category = defaultdict(lambda: [0, 0])
    for entry in report["results"]:
        by_category[entry["category"]][0] += entry["passed"]
        by_category[entry["category"]][1] += 1
    passed = sum(entry["passed"] for entry in report["results"])
    report["summary"] = {
        "passed": passed, "total": len(report["results"]),
        "by_category": {name: {"passed": p, "total": t} for name, (p, t) in by_category.items()},
        "modes": {mode: sum(e["mode"] == mode for e in report["results"]) for mode in sorted({e["mode"] for e in report["results"]})},
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    for name, (p, t) in by_category.items():
        print(f"  {name:<24} {p}/{t}")
    print(f"PASSED {passed}/{len(report['results'])}  modes={report['summary']['modes']}")
    print(f"Report: {out}", flush=True)


if __name__ == "__main__":
    main()
