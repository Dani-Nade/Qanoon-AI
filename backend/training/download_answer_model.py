"""Download a pinned instruction model for source-grounded explanations."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import model_info, snapshot_download

if __name__ == "__main__":
    model = "Qwen/Qwen3-4B-Instruct-2507"
    revision = model_info(model, timeout=30).sha
    print(f"Downloading {model} at {revision}", flush=True)
    snapshot = snapshot_download(
        model, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        max_workers=1,
    )
    target = ROOT / "models/answer-model.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"model": model, "revision": revision, "snapshot": snapshot}, indent=2), encoding="utf-8")
    print(f"Answer model ready: {target}", flush=True)
