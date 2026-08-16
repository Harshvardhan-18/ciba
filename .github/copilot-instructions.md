# Copilot Instructions — Agentic Creative Campaign Engine

A LangGraph "Creative Director" + asset planner + FLUX.2 Klein generation + hybrid evaluator (VLM + SigLIP + OCR) that turns a product + brand + brief into a 3-asset multi-channel campaign — not a `prompt -> image` tool. It's a portfolio/resume project, so the architecture (provider-abstracted generation, async 202/poll contract, stored attempt history) is the interview story; keep it clean enough to narrate.

## Never do these
- Never let the UX become `prompt -> generate image`; always go product + brand + brief -> planned campaign.
- Never block a request on image generation or LLM calls — use `POST -> 202` + background task + client polling `GET`.
- Never mix LangGraph reasoning with provider execution code; route all FLUX/VLM HTTP through `ImageGenerationProvider` / `VisionEvaluator`.
- Never overwrite a `GenerationAttempt`; each `CreativeAsset` keeps up to 3 attempts, each with its own `Evaluation`.
- Never let a high `overall_score` pass an asset that misses a `PASS_THRESHOLDS` hard constraint (especially `product_fidelity`) — use `evaluate_pass`.
- Never add Kafka, Qdrant, or Kubernetes to solve a V1 problem — prefer the boring Postgres/local-disk solution.

## Always do these
- Always keep image generation and vision scoring behind the abstract provider interfaces so providers stay swappable.
- Always default to mock providers in dev (`IMAGE_GENERATION_PROVIDER=mock`) and never call real/priced providers from tests or CI.
- Always use the 202 + poll contract for anything generation-related so Kafka can slot in later without changing the frontend.
- Always cap retries at 3 attempts per asset (enforce in the FastAPI layer, not graph state), treat infra failures separately, and land on `MANUAL_REVIEW` past the cap.

## Out of scope right now — do not add
Qdrant/vector retrieval, Kafka, Kubernetes/KEDA, S3/MinIO, OpenTelemetry/Prometheus/Grafana, channel-picker UI beyond the fixed 3 assets, billing/teams/multi-brand permissions.

## Repo map
- `CLAUDE_CODE_BRIEF.md` — frozen build brief / constitution (rules + V1 scope)
- `README.md` — project overview: pipeline, architecture story, stack, setup + config guide
- `pyproject.toml` — backend deps + pytest config (asyncio_mode=auto)
- `docker-compose.yml` — local Postgres 16 (ciba/ciba_dev/ciba_db)
- `alembic.ini` + `alembic/env.py` — Alembic config; env.py reads `DATABASE_URL` from `app.config`
- `alembic/versions/19baeac8e423_initial_schema.py` — initial migration (works around circular FKs)
- `app/config.py` — env settings; generation vars default to mock/dev-safe; includes `GROQ_*`, `GENERATE_PLACEMENTS` (all | subset for testing), `SQL_ECHO`
- `app/database.py` — async engine, `AsyncSessionLocal`, `get_db`, test-injectable `get_session_factory`
- `app/models.py` — frozen schema: User, Brand, Product, Campaign, CreativeConcept, CreativeAsset, GenerationAttempt, Evaluation
- `app/auth.py` — `get_current_user`: verifies NextAuth HS256 JWT, upserts `User` on `google_sub`
- `app/graph.py` — LangGraph (Director, Planner, gen/eval nodes) + provider interfaces/mocks + `PASS_THRESHOLDS` (fidelity 0.85)/`evaluate_pass` + Groq (default) & Gemini LLM calls; FLUX prompts are text-free (copy overlaid in post)
- `app/evaluation.py` — Checkpoint 4: `HybridVisionEvaluator` (VLM + SigLIP + OCR), frozen fidelity weights, lazy heavy deps; VLM signal switched by `VLM_PROVIDER` (`kaggle` worker | `gemini` API)
- `app/routes.py` — FastAPI contract: CRUD, campaign 202->poll, select-concept, assets poll, manual regenerate; generation runs as guarded detached tasks (serialized for the single-T4 worker) with in-process dedup; `infra_error` captured + logged + returned in the API
- `app/main.py` — FastAPI app, CORS, `/media` static mount for generated images, router at `/api/v1`, lifespan self-healing recovery loop (re-schedules stalled generation every 20s)
- `tests/conftest.py` — ensures `ciba_test` DB exists
- `tests/test_smoke.py` — Checkpoint 1: brand/product CRUD
- `tests/test_auth_integration.py` — real JWT -> user upsert (no dependency overrides)
- `tests/test_e2e_checkpoint2.py` — full campaign flow with mocked generation/LLM
- `tests/test_retry_loop.py` — 3-attempt generate/evaluate/diagnose loop with fake providers
- `tests/test_hybrid_scores.py` — hybrid fidelity weights + hard-constraint pass (no real models)
- `frontend/` — Next.js 16 + NextAuth v4: public marketing landing at `/` (hero over `public/background.png`, canvas particle wordmark + star field, pipeline, capability cards; redirects signed-in users to `/studio`), studio UI at `app/studio/` (brand/product setup with prefilled defaults, campaign brief, concept picker, live per-asset generation progress with attempt history + regenerate), `app/api/token/route.ts` mints HS256 JWTs, `app/lib/api.ts` client (`mediaUrl` handles Windows/`/` paths), reusable `--ld-*` landing design tokens in `globals.css` (colors sampled from `background.png`); `options.ts` uses custom HS256 JWS encode/decode so FastAPI's python-jose can verify
- `notebooks/flux-quality-test-working.ipynb` — standalone FLUX.2 Klein quality test (not integrated)
- `notebooks/kaggle_flux_worker.py` — Kaggle-side HTTP worker: loads FLUX.2 Klein, serves POST /generate for `RemoteFluxKaggleProvider`
- `notebooks/kaggle_eval_worker.py` — Kaggle-side eval worker: SmolVLM2 + SigLIP + RapidOCR, serves POST /evaluate for `HybridVisionEvaluator` (VLM_PROVIDER=kaggle)
- `notebooks/notebook_test.ipynb` — clean Kaggle notebook for the generation worker (ingest from dataset, start, tunnel, smoke test)
- `notebooks/eval_worker_test.ipynb` — clean Kaggle notebook for the eval worker (same flow, port 8001)
- `datasets/kaggle_flux_worker.py` — Kaggle-upload copy of the gen worker (single source of truth: `notebooks/`)
- `datasets/kaggle_eval_worker.py` — Kaggle-upload copy of the eval worker (single source of truth: `notebooks/`)
- `datasets/README.md` — Kaggle dataset upload + notebook attach + secrets guide + Checkpoint 4 verification steps

## Current state
> ⚠️ UPDATE ONLY THIS SECTION when a checkpoint lands.

All 5 checkpoints are implemented and validated end-to-end: foundation + auth + brand/product CRUD, the Director loop (Groq LLM) running campaign -> concepts -> select -> assets, real FLUX.2 Klein generation via the Kaggle worker, live hybrid evaluation (VLM + SigLIP + OCR) with the 3-attempt retry loop confirmed on real campaigns, and the Next.js studio + landing page. **20 tests pass** (`python -m pytest`).

Recurrent operational details (2026-08-16):
- Director LLM is **Groq** (`LLM_PROVIDER=groq`, `GROQ_API_KEY`/`GROQ_MODEL`) — Gemini kept as a swappable option.
- Generation is **serialized** (single-T4 worker) and **self-healing**: `_schedule_generation_loop` guards against double-scheduling; a lifespan recovery loop re-schedules stalled campaigns/assets every 20s.
- `PASS_THRESHOLDS["product_fidelity"]` is **0.85** (was 0.92): real FLUX scene-ads score ~0.81–0.92, so 0.92 sent every asset to `MANUAL_REVIEW` (owner decision 2026-08-15).
- FLUX prompts are **text-free** ("STRICT: no text/typography") — copy is overlaid in post; OCR no longer fails assets on garbled baked-in text.
- `GENERATE_PLACEMENTS=ig_feed` in `.env` restricts testing to one placement; `all` for the full 3.
- `infra_error` is captured, logged, and returned in the assets API; `SQL_ECHO` (default off) toggles SQL query logging; stored image paths are forward-slash normalized.
- Worker files live in `notebooks/` (single source of truth) with byte-identical `datasets/` copies — re-upload to Kaggle when changed. Gen worker has an inference lock; the eval worker is the hybrid scorer.
- Note: `next build` fails on this machine — a Next 16 + Node 24 framework bug in the `_global-error` prerender (reproduced on a pristine scaffold) — so use `next dev`.

## When in doubt
Prefer the boring Postgres-only solution over new infrastructure. Flag assumptions instead of guessing silently on anything touching the schema, auth, or retry logic.
