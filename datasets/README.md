# Kaggle upload bundle — FLUX.2 Klein workers

This folder is the ready-to-upload bundle for running the workers on Kaggle.
It contains two files:

- `kaggle_flux_worker.py` — the HTTP worker that serves FLUX.2 Klein
  generation (backend: `RemoteFluxKaggleProvider`, `KAGGLE_GATEWAY_URL`).
- `kaggle_eval_worker.py` — the HTTP worker that runs hybrid evaluation
  (VLM + SigLIP + OCR) for Checkpoint 4 (backend: `HybridVisionEvaluator` with
  `VLM_PROVIDER=kaggle`, `KAGGLE_EVAL_GATEWAY_URL`).

Both are copies of their single source of truth in `notebooks/` — if you
change a worker, re-copy it here before uploading.

They run in **two separate Kaggle notebooks/sessions** (each needs its own T4:
FLUX.2 Klein ~13GB fp16 cannot share a GPU with the eval models) and each gets
its own ngrok tunnel.

## 1. Upload both workers as a Kaggle dataset

1. Kaggle → **Datasets** → **New Dataset**.
2. Upload `kaggle_flux_worker.py` **and** `kaggle_eval_worker.py`
   (you can name the dataset anything, e.g. `ciba-workers`).
3. Publish it.

## 2. Attach the dataset to both notebooks

- `notebooks/notebook_test.ipynb` (generation worker) — the "Ingest worker"
  cell finds `kaggle_flux_worker.py` via glob.
- `notebooks/eval_worker_test.ipynb` (evaluation worker) — the "Ingest eval
  worker" cell finds `kaggle_eval_worker.py` via glob.

Dataset slug doesn't matter — any attached dataset containing the file works.

## 3. Add the two Kaggle Secrets

In **each** notebook: **Add-ons → Secrets** and add:

| Name | Value |
|------|-------|
| `HF_TOKEN` | your Hugging Face token (gated model `black-forest-labs/FLUX.2-klein-4B`) |
| `NGROK_AUTH_TOKEN` | your ngrok authtoken (from https://dashboard.ngrok.com) |

> Do **not** paste the ngrok token into notebook code — the clean notebooks
> read it from Secrets.

## 4. (Optional) Product-image dataset

Both workers condition on real product images from Kaggle dataset paths.
Attach a dataset with your product shots and set `PRODUCT_IMAGE` in the
notebooks' smoke-test cells to a `/kaggle/input/<dataset>/<file>` path. The
backend's `Product.product_images` should carry those same paths.

## 5. Backend env after both tunnels are up

```
IMAGE_GENERATION_PROVIDER=flux2_klein_kaggle
KAGGLE_GATEWAY_URL=<public url from the generation notebook's tunnel cell>

VISION_EVALUATOR_PROVIDER=hybrid
VLM_PROVIDER=kaggle
KAGGLE_EVAL_GATEWAY_URL=<public url from the eval notebook's tunnel cell>

# optional: KAGGLE_REQUEST_TIMEOUT_SECONDS=600
# optional: KAGGLE_API_KEY=<secret>   (workers don't verify it yet)
```

Switching to Gemini later = change `VLM_PROVIDER=gemini` and set
`GEMINI_API_KEY`; nothing else changes.

## Run order in each notebook

Setup → Secrets → Ingest worker → Start worker → Health → Tunnel → Smoke test.
Copy the printed PUBLIC URL into the matching backend env var.

## Verify Checkpoint 4 (kaggle-only path)

1. Eval notebook smoke test returns `status: 200` with all raw signals
   (`vlm_product_score`, `siglip_similarity`, `ocr_text_score`, etc.).
2. Backend has `VISION_EVALUATOR_PROVIDER=hybrid`, `VLM_PROVIDER=kaggle`,
   `KAGGLE_EVAL_GATEWAY_URL` set, `Product.product_images` pointing at the
   Kaggle dataset paths.
3. Run one campaign through the API: campaign → concepts → select → assets.
4. Confirm generated assets land `approved`/`manual_review` and `MEDIA_DIR`
   holds real PNGs, with `Evaluation` rows carrying non-mock scores and a
   `vlm_provider` of `kaggle`.
5. The real Checkpoint-4 acceptance test: force a failing first attempt (or
   just run with a hard product) and confirm the retry loop's
   `corrective_instruction` (built from the worker's `issues`) yields a
   higher score on attempt 2/3 than attempt 1 — the "system improved its
   output" demo.

