"""
FastAPI endpoint for QanoonAI.
Loads the fine-tuned model and answers legal queries.

Run:
    uvicorn api:app --reload --port 8000

Test:
    POST http://localhost:8000/query
    {"question": "What is the punishment for theft in Pakistan?"}
"""

from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL  = "Qwen/Qwen2-0.5B-Instruct"
LORA_DIR    = Path(__file__).parent / "training" / "qanoon-model"

app = FastAPI(title="QanoonAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model on startup ─────────────────────────────────────────────────────
tokenizer = None
model     = None


@app.on_event("startup")
def load_model() -> None:
    global tokenizer, model

    print("Loading QanoonAI model...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_DIR) if LORA_DIR.exists() else BASE_MODEL,
        trust_remote_code=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if LORA_DIR.exists():
        model = PeftModel.from_pretrained(base, str(LORA_DIR))
        print("Fine-tuned QanoonAI model loaded.")
    else:
        model = base
        print("WARNING: Fine-tuned model not found, using base model.")

    model.eval()


# ── Request / Response ────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    max_new_tokens: int = 512


class QueryResponse(BaseModel):
    question: str
    answer: str
    disclaimer: str = (
        "QanoonAI provides general legal information only. "
        "Consult a licensed lawyer for professional legal advice."
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    prompt = (
        f"<|im_start|>system\n"
        f"You are QanoonAI, a legal assistant specializing in Pakistani law.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{req.question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return QueryResponse(question=req.question, answer=answer)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}
