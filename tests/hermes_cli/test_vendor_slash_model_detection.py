"""Tests for vendor/model slash format detection in detect_provider_for_model.

Regression tests for the bug where typing `/model deepseek/deepseek-reasoner`
(without --provider) failed to switch the provider from 'anthropic' to 'deepseek',
sending the request to the wrong API endpoint (Anthropic instead of DeepSeek).

The root cause was that detect_provider_for_model() only checked bare model
names (e.g. "deepseek-reasoner") in _PROVIDER_MODELS, but never decomposed
"vendor/model" format into (provider, model) pairs.
"""

from unittest.mock import patch

from hermes_cli.models import detect_provider_for_model


class TestVendorSlashModelDetection:
    """Tests for Step 0.5: vendor/model slash format parsing."""

    def test_deepseek_slash_deepseek_reasoner(self):
        """deepseek/deepseek-reasoner should resolve to ('deepseek', 'deepseek-reasoner')."""
        result = detect_provider_for_model("deepseek/deepseek-reasoner", "anthropic")
        assert result is not None
        provider, model = result
        assert provider == "deepseek"
        assert model == "deepseek-reasoner"

    def test_deepseek_slash_deepseek_chat(self):
        """deepseek/deepseek-chat should resolve to ('deepseek', 'deepseek-chat')."""
        result = detect_provider_for_model("deepseek/deepseek-chat", "anthropic")
        assert result is not None
        provider, model = result
        assert provider == "deepseek"
        assert model == "deepseek-chat"

    def test_vendor_alias_resolution(self):
        """deep-seek/deepseek-reasoner should resolve via _PROVIDER_ALIASES."""
        result = detect_provider_for_model("deep-seek/deepseek-reasoner", "anthropic")
        assert result is not None
        provider, model = result
        assert provider == "deepseek"
        assert model == "deepseek-reasoner"

    def test_case_insensitive_match(self):
        """DeepSeek/DeepSeek-Reasoner should match case-insensitively."""
        result = detect_provider_for_model("DeepSeek/DeepSeek-Reasoner", "anthropic")
        assert result is not None
        provider, model = result
        assert provider == "deepseek"

    def test_unknown_vendor_returns_none_or_openrouter(self):
        """unknown-vendor/some-model should not match any built-in provider."""
        result = detect_provider_for_model("unknown-vendor/some-model", "anthropic")
        # Should either return None or fallback to OpenRouter — not stay on 'anthropic'
        if result is not None:
            assert result[0] != "anthropic"

    def test_unknown_model_in_known_vendor(self):
        """deepseek/nonexistent-model should NOT match (model not in catalog)."""
        result = detect_provider_for_model("deepseek/nonexistent-model", "anthropic")
        # The slash parsing should NOT match because the bare model isn't in the catalog
        # It may still match via OpenRouter or return None — but should not stay on anthropic
        # as the provider switch result from the slash parsing step
        if result is not None:
            # If it matched somewhere else (OpenRouter), that's fine
            pass
        # The main assertion: it didn't return ('deepseek', 'nonexistent-model')
        # from the slash parsing step alone
        if result is not None and result[0] == "deepseek":
            # This shouldn't happen — nonexistent-model is not in deepseek's catalog
            raise AssertionError(
                f"Should not match unknown model in deepseek catalog: {result}"
            )

    def test_no_slash_still_works(self):
        """Plain 'deepseek-reasoner' should still resolve to deepseek provider."""
        result = detect_provider_for_model("deepseek-reasoner", "anthropic")
        assert result is not None
        provider, model = result
        assert provider == "deepseek"

    def test_already_on_correct_provider(self):
        """deepseek/deepseek-reasoner when already on deepseek should check current catalog."""
        # When already on deepseek, the "current_models" check catches bare name
        # but not the slash format — our fix in Step 0.5 returns a switch even if
        # already on the same provider, which is handled by the caller.
        result = detect_provider_for_model("deepseek/deepseek-reasoner", "deepseek")
        # The result can be None (no switch needed) or ('deepseek', 'deepseek-reasoner')
        # Either is acceptable — what matters is it doesn't route to the wrong provider
        if result is not None:
            assert result[0] == "deepseek"

    def test_slash_with_empty_parts_ignored(self):
        """Edge case: '/model' or 'vendor/' should not crash."""
        result = detect_provider_for_model("/deepseek-reasoner", "anthropic")
        # Leading slash — should not trigger vendor/model parsing
        # (startswith("/") check in the code)
        if result is not None:
            # Fine as long as it doesn't crash
            pass

        result2 = detect_provider_for_model("deepseek/", "anthropic")
        # Empty model part — should not match
        if result2 is not None:
            assert result2[0] != "deepseek" or result2[1] != ""

    def test_anthropic_slash_model_not_misrouted(self):
        """anthropic/claude-opus-4-6 on non-anthropic provider should resolve correctly."""
        result = detect_provider_for_model("anthropic/claude-opus-4-6", "openrouter")
        # This should resolve — either to anthropic directly or stay on openrouter
        # The key test: it should NOT stay on a wrong provider
        if result is not None:
            assert result[0] in ("anthropic", "openrouter")
