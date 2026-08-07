"""
Unit tests for Checkpoint 4's hybrid scoring math and hard-constraint pass.

These test the pure aggregation (compute_hybrid_scores) and its interaction
with the frozen evaluate_pass thresholds. No real models are loaded — the
SigLIP/OCR/Gemini code paths are exercised only in live runs, never in tests
(consistent with "never call real/priced providers from tests or CI").
"""
import pytest

from app.evaluation import compute_hybrid_scores
from app.graph import evaluate_pass


def test_product_fidelity_uses_frozen_weights():
    # vlm 0.40 / siglip 0.35 / ocr 0.25 (frozen in the Evaluation schema).
    s = compute_hybrid_scores(1.0, 0.5, 0.5, 0.9, 0.9, 0.9, critical_text_error=False)
    expected_fidelity = 0.40 * 1.0 + 0.35 * 0.5 + 0.25 * 0.5  # 0.70
    assert s["product_fidelity"] == pytest.approx(expected_fidelity)
    assert s["overall_score"] == pytest.approx(
        0.45 * expected_fidelity + 0.20 * 0.9 + 0.20 * 0.9 + 0.15 * 0.9
    )


def test_low_product_fidelity_fails_despite_pretty_scores():
    # Beautiful ad (overall ~0.96) but product_fidelity 0.91 < 0.92 -> must FAIL.
    # This isolates the fidelity hard gate: overall is above threshold, fidelity
    # alone is below it.
    s = compute_hybrid_scores(0.90, 0.90, 0.94, 1.0, 1.0, 1.0, critical_text_error=False)
    assert s["product_fidelity"] == pytest.approx(0.91)
    assert s["overall_score"] > 0.90
    passed, reason = evaluate_pass(s)
    assert passed is False
    assert "product_fidelity" in (reason or "")


def test_critical_text_error_fails_regardless_of_score():
    s = compute_hybrid_scores(1.0, 1.0, 1.0, 0.99, 0.99, 0.99, critical_text_error=True)
    passed, reason = evaluate_pass(s)
    assert passed is False
    assert reason == "critical_text_error"


def test_passing_case():
    s = compute_hybrid_scores(0.95, 0.93, 1.0, 0.92, 0.91, 0.92, critical_text_error=False)
    assert s["product_fidelity"] >= 0.92
    passed, reason = evaluate_pass(s)
    assert passed is True
    assert reason is None


@pytest.mark.asyncio
async def test_hybrid_kaggle_requires_eval_gateway(monkeypatch):
    """VLM_PROVIDER=kaggle (the default) with no KAGGLE_EVAL_GATEWAY_URL must
    fail loudly instead of firing at nothing."""
    from app.config import settings
    from app.evaluation import HybridVisionEvaluator

    monkeypatch.setattr(settings, "VLM_PROVIDER", "kaggle")
    monkeypatch.setattr(settings, "KAGGLE_EVAL_GATEWAY_URL", "")

    ev = HybridVisionEvaluator()
    with pytest.raises(RuntimeError, match="KAGGLE_EVAL_GATEWAY_URL"):
        await ev.evaluate(["product.png"], "generated.png", {}, {})
