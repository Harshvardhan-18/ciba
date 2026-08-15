"""
Kaggle FLUX.2 Klein API worker — the tunneled HTTPS endpoint half of the V1
dev loop. It sits on a Kaggle T4, loads FLUX.2 Klein once, and bridges the
backend's RemoteFluxKaggleProvider to the pipeline:

    FastAPI backend (RemoteFluxKaggleProvider in app/graph.py)
        POST {KAGGLE_GATEWAY_URL}/generate
        { "product_images": [...], "reference_images": [...],
          "prompt": "...", "width": 1080, "height": 1350 }
            |
            |  ngrok/cloudflared tunnel  (KAGGLE_GATEWAY_URL)
            v
    This worker (Kaggle, owns the T4 + model)
        -> conditions on product/reference images
        -> returns { "image": "<base64 png>" }

How to run (Kaggle notebook cell, after your working FLUX cells):
    !pip install -q fastapi "uvicorn[standard]"
    !pip install -q "pillow<12.0,>=8.0"            # keep torchvision intact
    !cp /kaggle/working/kaggle_flux_worker.py .     # if you uploaded it
    !nohup python kaggle_flux_worker.py > worker.log 2>&1 &
    !cloudflared tunnel --url http://localhost:8000   # or: !ngrok http 8000
    # copy the public URL into the backend env: KAGGLE_GATEWAY_URL=<url>

Model/tuning notes are carried straight from
notebooks/flux-quality-test-working.ipynb (they were hard-won there):
  - ONE pipeline on cuda:0, fp16 — a T4 has no bf16 tensor cores.
  - Dimensions rounded down to a multiple of 16 (FLUX requirement).
  - enable_vae_slicing / tiling / attention_slicing to keep peak memory low.
  - KAGGLE_FLUX_RESOLUTION_SCALE=0.5 (etc.) if a T4 OOMs at full prod res.

Product images can be local paths on the Kaggle host (e.g.
/kaggle/input/<dataset>/product.jpg) or http(s) URLs — both are supported,
mirroring how the notebook already accepts PRODUCT_IMAGE_PATHS.
"""
from __future__ import annotations

import base64
import gc
import io
import os
import threading
import urllib.request
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Tunables (env vars, so the same script works on Kaggle without editing)
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("KAGGLE_FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
RESOLUTION_SCALE = float(os.environ.get("KAGGLE_FLUX_RESOLUTION_SCALE", "1.0"))
NUM_INFERENCE_STEPS = int(os.environ.get("KAGGLE_FLUX_STEPS", "4"))
GUIDANCE_SCALE = float(os.environ.get("KAGGLE_FLUX_GUIDANCE", "1.0"))
HOST = os.environ.get("KAGGLE_FLUX_HOST", "0.0.0.0")
PORT = int(os.environ.get("KAGGLE_FLUX_PORT", "8000"))

app = FastAPI(title="Kaggle FLUX.2 Klein worker", version="0.1.0")


class GenerateRequest(BaseModel):
    product_images: list[str] = []
    reference_images: list[str] = []
    prompt: str
    width: int = 1080
    height: int = 1350


class GenerateResponse(BaseModel):
    image: str  # base64-encoded PNG


# ---------------------------------------------------------------------------
# Pipeline (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_pipe = None
_pipe_lock = threading.Lock()
_infer_lock = threading.Lock()  # a T4 runs ONE FLUX generation at a time


def _load_pipeline():
    """Load FLUX.2 Klein 4B on one GPU (fp16 on CUDA). Mirrors the notebook."""
    from diffusers import Flux2KleinPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    gc.collect()
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        try:
            pipe.to("cuda:0")  # Klein 4B fits in ~13GB fp16 on one T4
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload(device="cuda:0")
    else:
        pipe.to(device)

    for method_name in ("enable_vae_slicing", "enable_vae_tiling", "enable_attention_slicing"):
        fn = getattr(pipe, method_name, None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
    return pipe


def get_pipe():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                _pipe = _load_pipeline()
    return _pipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scaled_dim(value: int, scale: float) -> int:
    """FLUX requires dimensions divisible by 16; also never below 16."""
    scaled = int(value * scale)
    return max(16, (scaled // 16) * 16)


def _load_image(source: str) -> Image.Image:
    if source.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(source).convert("RGB")


def _placeholder_image() -> Image.Image:
    return Image.new("RGB", (512, 512), color=(70, 110, 180))


def _load_reference_images(sources: list[str]) -> list[Image.Image]:
    images = []
    for src in sources:
        if not src:
            continue
        try:
            images.append(_load_image(src))
        except Exception as exc:
            print(f"[worker] skipping unreadable image {src!r}: {exc}")
    return images


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": _pipe is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    pipe = get_pipe()

    reference = _load_reference_images(req.product_images + req.reference_images)
    if not reference:
        print("[worker] no usable product/reference images; using placeholder")
        reference = [_placeholder_image()]

    width = _scaled_dim(req.width, RESOLUTION_SCALE)
    height = _scaled_dim(req.height, RESOLUTION_SCALE)

    # Cheap insurance against fragmentation between requests.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Serialize inference: concurrent pipe() calls on one T4 OOM the GPU and
    # kill the process mid-response ("server disconnected without sending a
    # response"). Lock so extra requests queue instead of crashing.
    try:
        with _infer_lock:
            output = pipe(
                prompt=req.prompt,
                image=reference,
                width=width,
                height=height,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    return GenerateResponse(image=base64.b64encode(buf.getvalue()).decode("ascii"))


if __name__ == "__main__":
    import uvicorn

    print(f"[worker] preloading {MODEL_ID} ...")
    get_pipe()  # load once up-front so the first /generate is fast
    uvicorn.run(app, host=HOST, port=PORT)
