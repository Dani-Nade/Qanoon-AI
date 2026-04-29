"""
Fine-tune Qwen2-0.5B-Instruct on Pakistani legal Q&A pairs using LoRA.

Requirements:
    pip install torch transformers peft datasets accelerate bitsandbytes

Run:
    python training/train.py

Output:
    training/qanoon-model/   <- fine-tuned LoRA adapter weights
"""

import json
import os
from pathlib import Path

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2-0.5B-Instruct"
QA_FILE      = Path(__file__).parent.parent / "data" / "training" / "qa_pairs.jsonl"
OUTPUT_DIR   = Path(__file__).parent / "qanoon-model"
MAX_LENGTH   = 256     # 512→256: attention is O(n²), biggest speed win
BATCH_SIZE   = 2       # fits with shorter seqs; fall back to 1 if OOM
GRAD_ACCUM   = 8       # effective batch = 16
EPOCHS       = 2       # sufficient for demo
LR           = 2e-4
# ──────────────────────────────────────────────────────────────────────────────


def load_qa_pairs() -> list[dict]:
    pairs = []
    with open(QA_FILE, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def format_prompt(example: dict) -> str:
    return (
        f"<|im_start|>system\n"
        f"You are QanoonAI, a legal assistant specializing in Pakistani law. "
        f"Answer questions accurately based on Pakistani statutes and legislation.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )


def tokenize(example: dict, tokenizer) -> dict:
    text = format_prompt(example)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load tokenizer & model ────────────────────────────────────────────────
    print(f"\nLoading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                        # rank — higher = more capacity, more VRAM
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading Q&A pairs...")
    raw_pairs = load_qa_pairs()
    print(f"  Total pairs: {len(raw_pairs)}")

    dataset = Dataset.from_list(raw_pairs)
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,    # Windows compatibility
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
