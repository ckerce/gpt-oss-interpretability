"""Tests for gossh.model_registry."""

import pytest

from gossh.model_registry import ModelArchSpec, get_arch_spec, list_supported_models


class TestModelArchSpec:
    def test_gpt_oss_20b_constants(self):
        spec = get_arch_spec("openai/gpt-oss-20b")
        assert spec.num_layers == 24
        assert spec.num_heads == 64
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 64
        assert spec.num_experts == 32
        assert spec.top_k == 4
        assert spec.quantization == "mxfp4"
        assert spec.vocab_size == 200019
        assert spec.max_seq_len == 8192

    def test_list_supported_models_includes_20b(self):
        models = list_supported_models()
        assert "openai/gpt-oss-20b" in models

    def test_120b_stub_not_in_supported(self):
        # 120b is a stub (num_layers=0) — not in list_supported_models
        models = list_supported_models()
        assert "openai/gpt-oss-120b" not in models

    def test_spec_is_frozen(self):
        spec = get_arch_spec("openai/gpt-oss-20b")
        with pytest.raises((AttributeError, TypeError)):
            spec.num_layers = 99

    def test_from_hf_config_classmethod_exists(self):
        assert hasattr(ModelArchSpec, "from_hf_config")
        assert callable(ModelArchSpec.from_hf_config)
