"""
Kaggle FLUX.2 Klein EVALUATION worker — the free-Kaggle half of Checkpoint 4.

Runs the hybrid evaluation signals (VLM + SigLIP + OCR) on a Kaggle T4 so the
backend can run Checkpoint 4 WITHOUT a Gemini subscription. When Gemini access
arrives, the backend just sets VLM_PROVIDER=gemini — no changes here.

    FastAPI backend (HybridVisionEvaluator, VLM_PROVIDER=kaggle)
        POST {KAGGLE_EVAL_GATEWAY_URL}/evaluate
        { product_images: [...], generated_image: "<base64 png>",
          asset_spec: {...}, brand_context: {...} }
            |
            |  ngrok tunnel (KAGGLE_EVAL_GATEWAY_URL)
            v
    This worker (a SEPARATE Kaggle notebook/session, its own T4)
        -> loads a small VLM (SmolVLM2), SigLIP, RapidOCR
        -> returns raw signals:
           { vlm_product_score, brand_consistency, composition_score,
             prompt_alignment, critical_text_error, issues,
             siglip_similarity, ocr_text_score, ocr_details }

Why a separate worker from the generation one: FLUX.2 Klein (~13GB fp16)
cannot share a T4 with a VLM, so generation and evaluation each get their own
disposable Kaggle session + tunnel.

Product images are resolved from local Kaggle dataset paths (the same ones the
generation worker uses) or http(s) URLs. The generated ad arrives as base64
because it lives on the backend.

Run (Kaggle notebook cell):
    !pip install -q fastapi "uvicorn[standard]" transformers "pillow<12.0,>=8.0" \
        rapidocr-onnxruntime "accelerate>=0.26.0"
    !nohup python kaggle_eval_worker.py > eval_worker.log 2>&1 &
    !cloudflared tunnel --url http://localhost:8001   # or: !ngrok http 8001
    # backend: KAGGLE_EVAL_GATEWAY_URL=<public url>
"""
from __future__ import annotations

import base64
import gc
import io
import json
import os
import re
import threading
import urllib.request

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Tunables (env vars)
# ---------------------------------------------------------------------------
VLM_MODEL_ID = os.environ.get("KAGGLE_EVAL_VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
SIGLIP_MODEL_ID = os.environ.get("KAGGLE_EVAL_SIGLIP_MODEL_ID", "google/siglip-so400m-patch14-384")
OCR_MIN_CONFIDENCE = float(os.environ.get("KAGGLE_EVAL_OCR_MIN_CONFIDENCE", "0.5"))
HOST = os.environ.get("KAGGLE_EVAL_HOST", "0.0.0.0")
PORT = int(os.environ.get("KAGGLE_EVAL_PORT", "8001"))

_HEADLINE_RE = re.compile(r'Headline text overlay:\s*"([^"]+)"', re.IGNORECASE)

app = FastAPI(title="Kaggle FLUX.2 Klein eval worker", version="0.1.0")


class EvaluateRequest(BaseModel):
    product_images: list[str] = []
    generated_image: str  # base64-encoded PNG
    asset_spec: dict = {}
    brand_context: dict = {}


class EvaluateResponse(BaseModel):
    vlm_product_score: float
    brand_consistency: float
    composition_score: float
    prompt_alignment: float
    critical_text_error: bool
    issues: list[str]
    siglip_similarity: float
    ocr_text_score: float
    ocr_details: dict


# ---------------------------------------------------------------------------
# Model loaders (lazy, loaded once, reused)
# ---------------------------------------------------------------------------

_vlm = None
_vlm_lock = threading.Lock()
_siglip = None
_siglip_lock = threading.Lock()
_ocr = None
_ocr_lock = threading.Lock()


def _get_vlm():
    global _vlm
    if _vlm is None:
        with _vlm_lock:
            if _vlm is None:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                model = AutoModelForImageTextToText.from_pretrained(
                    VLM_MODEL_ID, torch_dtype=dtype
                ).to(device).eval()
                processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
                _vlm = (model, processor, device)
                print(f"[eval worker] VLM loaded: {VLM_MODEL_ID} on {device}")
    return _vlm


def _get_siglip():
    global _siglip
    if _siglip is None:
        with _siglip_lock:
            if _siglip is None:
                from transformers import AutoProcessor, SiglipModel

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID).to(device).eval()
                processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
                _siglip = (model, processor, device)
                print(f"[eval worker] SigLIP loaded: {SIGLIP_MODEL_ID}")
    return _siglip


def _get_ocr():
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                from rapidocr_onnxruntime import RapidOCR

                _ocr = RapidOCR()
                print("[eval worker] RapidOCR loaded")
    return _ocr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(source: str) -> Image.Image:
    if source.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return Image.open(source).convert("RGB")


def load_optional_images(sources: list[str]) -> list[Image.Image]:
    images = []
    for src in sources:
        if not src:
            continue
        try:
            images.append(load_image(src))
        except Exception as exc:
            print(f"[eval worker] skipping unreadable image {src!r}: {exc}")
    return images


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"VLM did not return JSON: {text[:200]}")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# VLM scoring (SmolVLM2)
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


def _run_vlm(images: list[Image.Image], asset_spec: dict, brand_context: dict) -> dict:
    model, processor, device = _get_vlm()
    user_text = _VLM_SYSTEM_PROMPT + "\n\n" + _vlm_user_prompt(asset_spec, brand_context)
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img} for img in images]
                   + [{"type": "text", "text": user_text}],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    decoded = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    data = _extract_json(decoded)
    return {
        "vlm_product_score": float(data.get("vlm_product_score", 0.0)),
        "brand_consistency": float(data.get("brand_consistency", 0.0)),
        "composition_score": float(data.get("composition_score", 0.0)),
        "prompt_alignment": float(data.get("prompt_alignment", 0.0)),
        "critical_text_error": bool(data.get("critical_text_error", False)),
        "issues": [str(i) for i in data.get("issues", [])][:5],
    }


# ---------------------------------------------------------------------------
# SigLIP similarity
# ---------------------------------------------------------------------------

def _siglip_similarity(refs: list[Image.Image], gen: Image.Image) -> float:
    import torch.nn.functional as F

    if not refs:
        print("[eval worker] no product refs; siglip=0.0")
        return 0.0
    model, processor, device = _get_siglip()
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
# OCR
# ---------------------------------------------------------------------------

def _ocr_result(gen: Image.Image, asset_spec: dict) -> dict:
    engine = _get_ocr()
    result, _elapse = engine(np.array(gen.convert("RGB")))

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

    m = _HEADLINE_RE.search(asset_spec.get("generation_prompt", ""))
    expected = m.group(1).strip() if m else None
    headline_found = bool(expected) and any(expected.lower() in t.lower() for t in texts)

    if expected:
        ocr_text_score = 0.6 * avg_confidence + (0.4 if headline_found else 0.0)
        critical_text_error = not headline_found
    else:
        ocr_text_score = avg_confidence
        critical_text_error = bool(texts) and avg_confidence < OCR_MIN_CONFIDENCE

    return {
        "ocr_text_score": round(min(1.0, ocr_text_score), 4),
        "critical_text_error": critical_text_error,
        "detected_texts": texts,
        "avg_confidence": round(avg_confidence, 4),
        "expected_headline": expected,
        "headline_found": headline_found,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vlm_loaded": _vlm is not None,
        "siglip_loaded": _siglip is not None,
        "ocr_loaded": _ocr is not None,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    refs = load_optional_images(req.product_images)
    try:
        gen = Image.open(io.BytesIO(base64.b64decode(req.generated_image))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid generated_image: {exc}") from exc

    vlm = _run_vlm(refs, req.asset_spec, req.brand_context)
    siglip = _siglip_similarity(refs, gen)
    ocr = _ocr_result(gen, req.asset_spec)

    return EvaluateResponse(
        vlm_product_score=vlm["vlm_product_score"],
        brand_consistency=vlm["brand_consistency"],
        composition_score=vlm["composition_score"],
        prompt_alignment=vlm["prompt_alignment"],
        critical_text_error=vlm["critical_text_error"] or ocr["critical_text_error"],
        issues=vlm["issues"],
        siglip_similarity=siglip,
        ocr_text_score=ocr["ocr_text_score"],
        ocr_details=ocr,
    )


if __name__ == "__main__":
    import uvicorn

    print(f"[eval worker] preloading {VLM_MODEL_ID} ...")
    _get_vlm()
    _get_siglip()
    _get_ocr()
    uvicorn.run(app, host=HOST, port=PORT)
