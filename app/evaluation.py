"""
Checkpoint 4 — real hybrid evaluation behind the VisionEvaluator interface.

HybridVisionEvaluator combines three signals, matching the frozen Evaluation
schema exactly:
  - VLM:               vlm_product_score, brand_consistency, composition_score,
                       prompt_alignment, critical_text_error + issues (diagnosis)
  - SigLIP:            siglip_similarity (how well the product identity survives)
  - OCR:               ocr_text_score + critical_text_error (headline legibility)

The VLM signal is provider-swappable via settings.VLM_PROVIDER (one config
change, same contract as image generation):
  - "kaggle" (default, free): the whole evaluation runs on the Kaggle eval
    worker (notebooks/kaggle_eval_worker.py) via KAGGLE_EVAL_GATEWAY_URL —
    needed because the product images live in a Kaggle dataset, and there's
    no paid VLM subscription yet.
  - "gemini": the VLM runs via the Gemini API; SigLIP + OCR stay local.

product_fidelity is the frozen weighted blend: 40% VLM / 35% SigLIP / 25% OCR.
The hard-constraint pass logic stays in app.graph.evaluate_pass (product_fidelity
>= 0.92, etc.) — nothing here can average a broken product into a pass.

Heavy deps (torch/transformers for SigLIP, rapidocr_onnxruntime for OCR, the
Gemini SDK) are imported lazily inside the functions that need them, so
importing this module is cheap and mock-only tests/CI never pull them.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import threading
import urllib.request

import httpx
from PIL import Image

from app.config import settings
from app.graph import VisionEvaluator, evaluate_pass

logger = logging.getLogger(__name__)

# Frozen weighted blend (see Evaluation.product_fidelity schema comment).
FIDELITY_WEIGHTS = {"vlm": 0.40, "siglip": 0.35, "ocr": 0.25}
# overall = 45% fidelity + 20% brand + 20% composition + 15% alignment.
_OVERALL_WEIGHTS = {"fidelity": 0.45, "brand": 0.20, "composition": 0.20, "alignment": 0.15}

_VLM_MODEL = "gemini-1.5-flash"
_HEADLINE_RE = re.compile(r'Headline text overlay:\s*"([^"]+)"', re.IGNORECASE)


def compute_hybrid_scores(
    vlm_product_score: float,
    siglip_similarity: float,
    ocr_text_score: float,
    brand_consistency: float,
    composition_score: float,
    prompt_alignment: float,
    critical_text_error: bool,
) -> dict:
    """Pure aggregation — unit-testable without any models."""
    product_fidelity = (
        FIDELITY_WEIGHTS["vlm"] * vlm_product_score
        + FIDELITY_WEIGHTS["siglip"] * siglip_similarity
        + FIDELITY_WEIGHTS["ocr"] * ocr_text_score
    )
    overall_score = (
        _OVERALL_WEIGHTS["fidelity"] * product_fidelity
        + _OVERALL_WEIGHTS["brand"] * brand_consistency
        + _OVERALL_WEIGHTS["composition"] * composition_score
        + _OVERALL_WEIGHTS["alignment"] * prompt_alignment
    )
    return {
        "vlm_product_score": vlm_product_score,
        "siglip_similarity": siglip_similarity,
        "ocr_text_score": ocr_text_score,
        "product_fidelity": product_fidelity,
        "brand_consistency": brand_consistency,
        "composition_score": composition_score,
        "prompt_alignment": prompt_alignment,
        "overall_score": overall_score,
        "critical_text_error": critical_text_error,
    }


# ---------------------------------------------------------------------------
# Image loading (local path or http(s) URL)
# ---------------------------------------------------------------------------

def load_image(source: str) -> Image.Image:
    if source.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return Image.open(source).convert("RGB")


# ---------------------------------------------------------------------------
# SigLIP (local, lazy-loaded)
# ---------------------------------------------------------------------------

_siglip = None
_siglip_lock = threading.Lock()


def _get_siglip():
    global _siglip
    if _siglip is None:
        with _siglip_lock:
            if _siglip is None:
                import torch
                from transformers import AutoProcessor, SiglipModel

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = SiglipModel.from_pretrained(settings.SIGLIP_MODEL_ID).to(device).eval()
                processor = AutoProcessor.from_pretrained(settings.SIGLIP_MODEL_ID)
                _siglip = (model, processor, device)
    return _siglip


def _siglip_similarity(original_images: list[str], generated_image: str) -> float:
    """Average cosine similarity (mapped to [0,1]) between product refs and the ad."""
    import torch
    import torch.nn.functional as F

    model, processor, device = _get_siglip()

    refs = []
    for src in original_images:
        try:
            refs.append(load_image(src))
        except Exception as exc:
            logger.warning("hybrid evaluator: skipping unreadable product image %s: %s", src, exc)
    if not refs:
        logger.warning("hybrid evaluator: no readable product images; siglip=0.0")
        return 0.0
    gen = load_image(generated_image)

    ref_inputs = processor(images=refs, return_tensors="pt").to(device)
    gen_inputs = processor(images=[gen], return_tensors="pt").to(device)
    with torch.no_grad():
        ref_feat = model.get_image_features(**ref_inputs)
        gen_feat = model.get_image_features(**gen_inputs)
    ref_feat = F.normalize(ref_feat, dim=-1)
    gen_feat = F.normalize(gen_feat, dim=-1)
    sim = float((ref_feat @ gen_feat.T).mean().item())
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


# ---------------------------------------------------------------------------
# OCR (local, lazy-loaded via rapidocr_onnxruntime)
# ---------------------------------------------------------------------------

_ocr = None
_ocr_lock = threading.Lock()


def _get_ocr():
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                from rapidocr_onnxruntime import RapidOCR

                _ocr = RapidOCR()
    return _ocr


def _ocr_result(generated_image: str, asset_spec: dict) -> dict:
    """OCR the ad; score text legibility + whether the expected headline rendered."""
    engine = _get_ocr()
    result, _elapse = engine(generated_image)

    texts: list[str] = []
    confs: list[float] = []
    if result:
        for item in result:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
                confs.append(float(item.get("score", 0.0)))
            else:  # older format: [box, text, score]
                texts.append(str(item[1]))
                confs.append(float(item[2]))
    avg_confidence = (sum(confs) / len(confs)) if confs else 0.0

    # Expected headline, derived from the spec's frozen prompt format.
    m = _HEADLINE_RE.search(asset_spec.get("generation_prompt", ""))
    expected = m.group(1).strip() if m else None
    headline_found = bool(expected) and any(expected.lower() in t.lower() for t in texts)

    if expected:
        ocr_text_score = 0.6 * avg_confidence + (0.4 if headline_found else 0.0)
        critical_text_error = not headline_found
    else:
        ocr_text_score = avg_confidence
        critical_text_error = bool(texts) and avg_confidence < settings.OCR_MIN_CONFIDENCE

    return {
        "ocr_text_score": round(min(1.0, ocr_text_score), 4),
        "critical_text_error": critical_text_error,
        "detected_texts": texts,
        "avg_confidence": round(avg_confidence, 4),
        "expected_headline": expected,
        "headline_found": headline_found,
    }


# ---------------------------------------------------------------------------
# VLM signal sources — swapped via settings.VLM_PROVIDER (one config change)
# ---------------------------------------------------------------------------

_VLM_SYSTEM_PROMPT = """\
You are a strict creative-director evaluator. Given the ORIGINAL PRODUCT IMAGE(S)
and the GENERATED AD IMAGE, score the generated ad on product fidelity and craft.

Respond with ONLY a JSON object (no markdown, no prose):
{
  "vlm_product_score": 0.0..1.0,
  "brand_consistency": 0.0..1.0,
  "composition_score": 0.0..1.0,
  "prompt_alignment": 0.0..1.0,
  "critical_text_error": false,
  "issues": ["1-3 short concrete issues"]
}

Rules:
- vlm_product_score = how faithfully the generated ad preserves the product's
  identity (shape, colors, materials, details) vs the original product image.
  Below 0.92 is a FAIL regardless of every other score.
- critical_text_error = true if headline/overlay text is garbled, misspelled,
  or missing.
- Keep issues short and actionable — they feed the next attempt's correction.
"""


def _vlm_user_prompt(asset_spec: dict, brand_context: dict) -> str:
    return (
        "ASSET SPEC:\n"
        f"  placement={asset_spec.get('placement')} {asset_spec.get('width')}x{asset_spec.get('height')}\n"
        f"  prompt={asset_spec.get('generation_prompt', '')}\n"
        "BRAND CONTEXT:\n"
        f"  {brand_context or {}}"
    )


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"VLM did not return JSON: {text[:200]}")
    return json.loads(text[start:end + 1])


async def _gemini_vlm_score(
    original_images: list[str],
    generated_image: str,
    asset_spec: dict,
    brand_context: dict,
) -> dict:
    """VLM signal via the Gemini Flash API (requires GEMINI_API_KEY)."""
    import google.generativeai as genai
    from google.generativeai import types as genai_types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is required for VLM_PROVIDER=gemini")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=_VLM_MODEL,
        system_instruction=_VLM_SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    parts = []
    for src in original_images:
        try:
            buf = io.BytesIO()
            load_image(src).save(buf, format="PNG")
            parts.append(genai_types.Part.from_data(data=buf.getvalue(), mime_type="image/png"))
        except Exception as exc:
            logger.warning("hybrid evaluator: skipping unreadable product image %s: %s", src, exc)
    gen_buf = io.BytesIO()
    load_image(generated_image).save(gen_buf, format="PNG")
    parts.append(genai_types.Part.from_data(data=gen_buf.getvalue(), mime_type="image/png"))
    parts.append(genai_types.Part.from_text(_vlm_user_prompt(asset_spec, brand_context)))

    resp = await model.generate_content_async(parts)
    data = _extract_json(resp.text)

    return {
        "vlm_product_score": float(data.get("vlm_product_score", 0.0)),
        "brand_consistency": float(data.get("brand_consistency", 0.0)),
        "composition_score": float(data.get("composition_score", 0.0)),
        "prompt_alignment": float(data.get("prompt_alignment", 0.0)),
        "critical_text_error": bool(data.get("critical_text_error", False)),
        "issues": [str(i) for i in data.get("issues", [])][:5],
        "raw": resp.text,
    }


async def _kaggle_eval_signals(
    original_images: list[str],
    generated_image: str,
    asset_spec: dict,
    brand_context: dict,
) -> dict:
    """
    VLM + SigLIP + OCR signals from the Kaggle eval worker (VLM_PROVIDER=kaggle).

    The generated ad is sent as base64 (it lives on the backend); the product
    image paths are passed through so the worker can resolve them from the
    Kaggle dataset (or fetch them as URLs). Returns the worker's raw signal
    dict — the frozen aggregation/pass math stays here on the backend.
    """
    base_url = settings.KAGGLE_EVAL_GATEWAY_URL.rstrip("/")
    if not base_url:
        raise RuntimeError(
            "VLM_PROVIDER=kaggle requires KAGGLE_EVAL_GATEWAY_URL set to the Kaggle "
            "eval worker tunnel (notebooks/kaggle_eval_worker.py), or use "
            "VLM_PROVIDER=gemini / VISION_EVALUATOR_PROVIDER=mock."
        )

    buf = io.BytesIO()
    load_image(generated_image).save(buf, format="PNG")
    payload = {
        "product_images": original_images,
        "generated_image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "asset_spec": asset_spec,
        "brand_context": brand_context,
    }

    async with httpx.AsyncClient(timeout=settings.KAGGLE_REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.post(f"{base_url}/evaluate", json=payload)
        resp.raise_for_status()
        data = resp.json()

    required = {
        "vlm_product_score", "siglip_similarity", "ocr_text_score",
        "brand_consistency", "composition_score", "prompt_alignment",
        "critical_text_error",
    }
    missing = required - set(data)
    if missing:
        raise RuntimeError(f"Kaggle eval worker response missing fields: {missing}")
    data["issues"] = [str(i) for i in data.get("issues", [])][:5]
    return data


# ---------------------------------------------------------------------------
# Hybrid evaluator
# ---------------------------------------------------------------------------

class HybridVisionEvaluator(VisionEvaluator):
    """
    Real VLM + SigLIP + OCR hybrid behind the frozen VisionEvaluator interface.
    The VLM signal source is picked by settings.VLM_PROVIDER:
      - "kaggle" (default): every signal comes from the Kaggle eval worker
        (product refs live in a Kaggle dataset; no paid subscription needed).
      - "gemini": Gemini VLM; SigLIP + OCR run locally in a thread executor.
    Any per-signal failure degrades the relevant score honestly (never a
    silent high score); a full signal failure raises -> INFRA_FAILED.
    """

    async def evaluate(
        self,
        original_product_images: list[str],
        generated_image: str,
        asset_spec: dict,
        brand_context: dict,
    ) -> dict:
        provider = settings.VLM_PROVIDER.strip().lower()

        if provider == "kaggle":
            raw = await _kaggle_eval_signals(
                original_product_images, generated_image, asset_spec, brand_context
            )
            scores = compute_hybrid_scores(
                vlm_product_score=raw["vlm_product_score"],
                siglip_similarity=raw["siglip_similarity"],
                ocr_text_score=raw["ocr_text_score"],
                brand_consistency=raw["brand_consistency"],
                composition_score=raw["composition_score"],
                prompt_alignment=raw["prompt_alignment"],
                critical_text_error=raw["critical_text_error"],
            )
            raw_response = {"vlm": {"issues": raw.get("issues", []), "raw": raw}}
        elif provider == "gemini":
            vlm = await _gemini_vlm_score(
                original_product_images, generated_image, asset_spec, brand_context
            )
            siglip = await asyncio.to_thread(_siglip_similarity, original_product_images, generated_image)
            ocr = await asyncio.to_thread(_ocr_result, generated_image, asset_spec)
            scores = compute_hybrid_scores(
                vlm_product_score=vlm["vlm_product_score"],
                siglip_similarity=siglip,
                ocr_text_score=ocr["ocr_text_score"],
                brand_consistency=vlm["brand_consistency"],
                composition_score=vlm["composition_score"],
                prompt_alignment=vlm["prompt_alignment"],
                critical_text_error=vlm["critical_text_error"] or ocr["critical_text_error"],
            )
            raw_response = {
                "vlm": {"issues": vlm["issues"], "raw": vlm["raw"]},
                "siglip": siglip,
                "ocr": ocr,
            }
        else:
            raise RuntimeError(
                f"Unknown VLM_PROVIDER: {provider!r} (use 'kaggle' or 'gemini')"
            )

        passed, reason = evaluate_pass(scores)
        scores["passed"] = passed
        scores["failure_reason"] = reason
        scores["vlm_provider"] = provider
        scores["raw_response"] = raw_response
        return scores
