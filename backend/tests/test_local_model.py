import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from qanoon_ai.core.config import settings
from qanoon_ai.llm.client import adapter_complete, LLMResult, LocalLLMClient, validate_quotation
from qanoon_ai.llm.prompts import ABSTENTION
from qanoon_ai.retrieval.models import EvidenceChunk
from qanoon_ai.retrieval.service import LegalAnswerService
from qanoon_ai.schemas.query import QueryRequest
from training.prepare import encode_example, prepare_examples


class LocalModelTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("QANOON_TEST_LOCAL_MODEL") == "1", "Optional saved-model integration test")
    def test_saved_model_loads_without_network_access(self):
        client = LocalLLMClient(settings.trained_model_dir, settings.root_dir / "data/cache/huggingface/hub")
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected network access")) as network:
            self.assertTrue(client.ready)
            network.assert_not_called()

    def test_empty_or_partial_model_is_not_ready(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(adapter_complete(path))
            (path / "adapter_model.safetensors").write_bytes(b"incomplete")
            self.assertFalse(adapter_complete(path))

    def test_quotes_must_match_the_numbered_source_without_extra_claims(self):
        excerpt = "A civil servant shall perform the duties assigned to the post."
        self.assertTrue(validate_quotation(f'"{excerpt}" [1]', [excerpt]))
        self.assertTrue(validate_quotation(ABSTENTION, []))
        self.assertFalse(validate_quotation(f'"{excerpt}" [2]', [excerpt]))
        self.assertFalse(validate_quotation('"The penalty is ten years in prison." [1]', [excerpt]))
        self.assertFalse(validate_quotation(f'Invented claim. "{excerpt}" [1]', [excerpt]))

    def test_preparation_deduplicates_and_holds_out_whole_documents(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            rows = [
                {"output": f"According to Example Act {index}:\n\n" +
                 f"The provision for document {index} establishes the functions of the authority. " * 8}
                for index in range(100)
            ]
            rows.append(rows[0])
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            train, evaluation, report = prepare_examples(path)
            self.assertTrue(train and evaluation)
            self.assertEqual(report["duplicate_excerpts_removed"], 1)
            self.assertFalse({row["document"] for row in train} & {row["document"] for row in evaluation})
            for row in train + evaluation:
                if row["answer"] != ABSTENTION:
                    quote = row["answer"][1:-5]
                    self.assertIn(quote, row["messages"][-1]["content"])

    def test_prompt_tokens_are_masked_and_long_examples_are_dropped(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3, 4, 5] if len(messages) == 3 else [1, 2, 3]

        example = {"messages": [{}, {}], "answer": "answer"}
        encoded = encode_example(example, Tokenizer(), 8)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 4, 5])
        self.assertEqual(encode_example(example, Tokenizer(), 4)["input_ids"], [])

    def test_attention_without_gqa_matches_sdpa(self):
        import torch
        from transformers import AttentionInterface
        from qanoon_ai.llm.attention import register_attention

        attention = AttentionInterface()[register_attention()]
        query, key, value = torch.randn(1, 8, 5, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
        output, _ = attention(Mock(num_key_value_groups=4, is_causal=True), query, key, value, None)
        torch.testing.assert_close(output, expected.transpose(1, 2))

    def test_cuda_failures_mark_answer_model_unavailable_but_oom_does_not(self):
        import torch
        from qanoon_ai.llm.answer import LocalAnswerClient

        class Inputs(dict):
            input_ids = torch.ones(1, 3, dtype=torch.long)

            def to(self, device):
                return self

        client = LocalAnswerClient(Path("unused"))
        client._tokenizer = Mock(return_value=Inputs(), eos_token_id=0)
        client._model = Mock(device="cpu")
        client._model.generate.side_effect = torch.OutOfMemoryError("CUDA out of memory")
        with self.assertRaises(torch.OutOfMemoryError):
            client._generate([])
        self.assertFalse(client._failed)
        client._model.generate.side_effect = RuntimeError("CUDA error: out of memory")
        with self.assertRaises(RuntimeError):
            client._generate([])
        self.assertTrue(client._failed)
        self.assertFalse(client.ready)

    def test_service_uses_model_and_falls_back_when_generation_fails(self):
        service = LegalAnswerService(replace(settings, retrieval_mode="sqlite", llm_mode="disabled"))
        service.sqlite_retriever = Mock(ready=True)
        service.sqlite_retriever.search.return_value = [EvidenceChunk(
            chunk_id="test", text="A civil servant shall perform the duties assigned to the post.",
            source_file="example.pdf", title="Example Act", year="2024", score=1.0,
        )]
        service.llm = Mock(ready=True)
        service.llm.answer.return_value = LLMResult('"A civil servant shall perform the duties assigned to the post." [1]', "test")
        answer = service.answer(QueryRequest(question="civil servants"))
        self.assertEqual(answer.answer, service.llm.answer.return_value.text)
        self.assertTrue(answer.model_ready)
        service.llm.answer.side_effect = ValueError("Unverified output")
        answer = service.answer(QueryRequest(question="civil servants"))
        self.assertIn("duties assigned", answer.answer)
        self.assertTrue(any("could not be verified" in warning for warning in answer.warnings))


if __name__ == "__main__":
    unittest.main()
