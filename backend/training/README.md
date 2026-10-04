# Local Qanoon model

This run creates a **source-quotation prototype**, using Qwen2.5-0.5B-Instruct
and a LoRA adapter. The existing 16,447 records contain automatically generated
questions and raw legal excerpts, not expert-reviewed legal answers. Preparation
replaces unsupported question/answer pairings with quotation tasks, removes
duplicate excerpts, adds no-source abstention examples, and holds out entire
document titles for evaluation. This does not establish legal accuracy or Urdu
answer-generation quality.

From the repository root, install CUDA-enabled PyTorch for your GPU, then:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-local-model.txt
.\backend\.venv\Scripts\python.exe -u backend/training/train.py
```

Defaults: two epochs, batch size 2, gradient accumulation 8, seed 42, maximum
768 tokens, assistant-only loss, and checkpoints/evaluation every 200 steps.
Overlength examples are skipped rather than
training a truncated answer. The base model revision is recorded and reused for
inference. Checkpoints and JSON progress logs are saved to `training/runs/`.
The final adapter and tokenizer are saved to `training/qanoon-model/` only after
held-out loss improves over the untrained adapter baseline. Evaluation loss
measures this quotation task, not legal expertise.

Useful overrides: `--epochs 1`, `--batch-size 2`, `--output <new-directory>`,
`--resume <checkpoint-directory>`. A completed adapter is never overwritten.

The backend defaults to `QANOON_LLM_MODE=auto`: it loads a completed adapter from
the local cache when available. `disabled` keeps the extraction-only behavior;
`local` explicitly enables the adapter. Base-model downloads are not allowed
during API requests. Restart the API after installing its model dependencies.

Generated quotations must be exact substrings of their numbered citations.
Malformed or invented quotations fall back to the retrieved extracts with an
explicit warning. This check verifies quotation fidelity, not relevance or the
current legal status of a source. Questions with no retrieved sources bypass
generation. The index and model are separate: training does not expand the
26-document development search index.
