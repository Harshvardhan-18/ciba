# Kaggle upload bundle — FLUX.2 Klein worker

This folder is the ready-to-upload bundle for running the worker on Kaggle.
It contains one file:

- `kaggle_flux_worker.py` — the HTTP worker that serves FLUX.2 Klein and
  connects the backend (`RemoteFluxKaggleProvider`) to a Kaggle T4. It is a
  copy of `notebooks/kaggle_flux_worker.py` (single source of truth lives
  there — if you change the worker, re-copy it here before uploading).

## 1. Upload the worker as a Kaggle dataset

1. Kaggle → **Datasets** → **New Dataset**.
2. Upload `kaggle_flux_worker.py` (you can name the dataset anything, e.g. `ciba-worker`).
3. Publish it.

## 2. Attach it to the worker notebook

In `notebooks/notebook_test.ipynb` (the notebook you run on Kaggle):

- **Add input → Datasets** → pick the dataset above. The "Ingest worker" cell
  locates the file automatically via `glob.glob("/kaggle/input/*/kaggle_flux_worker.py")`,
  so the dataset slug doesn't matter — any attached dataset containing the file works.

## 3. Add the two Kaggle Secrets

In the notebook: **Add-ons → Secrets** and add:

| Name | Value |
|------|-------|
| `HF_TOKEN` | your Hugging Face token (gated model `black-forest-labs/FLUX.2-klein-4B`) |
| `NGROK_AUTH_TOKEN` | your ngrok authtoken (from https://dashboard.ngrok.com) |

> Do **not** paste the ngrok token into notebook code — the clean notebook
> reads it from Secrets.

## 4. (Optional) Product-image dataset

The worker conditions on real product images. Either:

- attach a dataset containing your product shots and set `PRODUCT_IMAGE` in
  the notebook's smoke-test cell to a `/kaggle/input/<dataset>/<file>` path, or
- point the backend's `Product.product_images` at those same paths so the
  requests carry them.

## 5. Backend env after the tunnel is up

```
IMAGE_GENERATION_PROVIDER=flux2_klein_kaggle
KAGGLE_GATEWAY_URL=<public url from the notebook's tunnel cell>
# optional: KAGGLE_REQUEST_TIMEOUT_SECONDS=600
# optional: KAGGLE_API_KEY=<secret>   (worker doesn't verify it yet)
```

## Run order in the notebook

Setup → Secrets → Ingest worker → Start worker → Health → Tunnel → Smoke test.
Copy the printed PUBLIC URL into `KAGGLE_GATEWAY_URL`.
