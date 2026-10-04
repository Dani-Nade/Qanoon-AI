from unittest.mock import Mock

import pytest

from training import smoke_api


def test_smoke_test_accepts_current_health_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke_api, "BACKEND", tmp_path)
    (tmp_path / "training/qanoon-model").mkdir(parents=True)
    monkeypatch.setattr(smoke_api.requests, "get", Mock(return_value=Mock(json=lambda: {
        "llm_mode": "quotation", "model_ready": True, "trained_adapter_available": True,
    })))
    post = Mock(return_value=Mock(json=lambda: {
        "answer": "No evidence.", "model_ready": True, "citations": [], "warnings": [],
    }))
    monkeypatch.setattr(smoke_api.requests, "post", post)
    smoke_api.main()
    assert post.call_count == 4
    assert (tmp_path / "training/qanoon-model/api-evaluation.json").is_file()


def test_smoke_test_reports_disabled_model_without_key_error(monkeypatch):
    monkeypatch.setattr(smoke_api.requests, "get", Mock(return_value=Mock(json=lambda: {
        "llm_mode": "extracts", "model_ready": False, "trained_adapter_available": True,
    })))
    with pytest.raises(RuntimeError, match="QANOON_LLM_MODE=quotation"):
        smoke_api.main()
