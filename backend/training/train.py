"""Train a local LoRA quotation model: python -m training.train (from backend)."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from training.prepare import encode_example, prepare_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/qa_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=BACKEND / "training/qanoon-model")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from huggingface_hub import model_info
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Trainer, TrainerCallback, TrainingArguments, set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Install CUDA-enabled PyTorch before training.")
    if (args.output / "adapter_model.safetensors").exists():
        raise FileExistsError(f"A trained adapter already exists at {args.output}; choose a new --output")
    set_seed(42)
    run = BACKEND / "training/runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)
    train, evaluation, report = prepare_examples(args.data)
    print(json.dumps(report, indent=2), flush=True)
    revision = model_info(args.base_model).sha
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def dataset(rows):
        encoded = [encode_example(row, tokenizer, args.max_length) for row in rows]
        valid = [row for row in encoded if row["input_ids"]]
        if not valid:
            raise ValueError("No examples fit max-length")
        return Dataset.from_list(valid), len(encoded) - len(valid)

    train_ds, train_dropped = dataset(train)
    eval_ds, eval_dropped = dataset(evaluation)
    report.update({
        "base_model": args.base_model, "base_revision": revision,
        "train_rows_used": len(train_ds), "eval_rows_used": len(eval_ds),
        "overlength_rows_dropped": train_dropped + eval_dropped,
        "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "run_dir": str(run), "epochs": args.epochs, "seed": 42,
        "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
    })
    (run / "data-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (run / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for row in evaluation:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=revision,
        dtype=torch.bfloat16 if bf16 else torch.float32, attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    model.print_trainable_parameters()

    started = time.monotonic()

    class Progress(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            entry = {
                "step": state.global_step, "total_steps": state.max_steps,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                **(logs or {}),
            }
            with (run / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(run), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=1e-4, bf16=bf16, fp16=False, optim="adamw_torch",
            logging_steps=10, logging_first_step=True,
            eval_strategy="steps", eval_steps=200,
            save_strategy="steps", save_steps=200, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, warmup_ratio=0.05, lr_scheduler_type="cosine",
            report_to="none", dataloader_num_workers=0, seed=42,
            prediction_loss_only=True, disable_tqdm=True,
        ),
        train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8), callbacks=[Progress()],
    )
    baseline = trainer.evaluate()
    print(f"Baseline evaluation: {baseline}", flush=True)
    try:
        result = trainer.train(resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        # Preserve resumable optimizer/RNG state if the user interrupts a long run.
        trainer._save_checkpoint(model, trial=None)
        print(f"Training interrupted; resumable checkpoint saved under {run}", flush=True)
        raise
    evaluation_metrics = trainer.evaluate()
    report.update({"baseline": baseline, "training": result.metrics, "evaluation": evaluation_metrics})
    (run / "training-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not math.isfinite(evaluation_metrics["eval_loss"]) or evaluation_metrics["eval_loss"] >= baseline["eval_loss"]:
        raise RuntimeError("Held-out loss did not improve; checkpoints retained, adapter not published")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TRAINING COMPLETE: {args.output}", flush=True)
    print(json.dumps(evaluation_metrics), flush=True)


if __name__ == "__main__":
    main()
