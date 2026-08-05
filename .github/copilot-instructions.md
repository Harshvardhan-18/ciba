# Copilot Instructions — Agentic Creative Campaign Engine

A LangGraph "Creative Director" + asset planner + FLUX.2 Klein generation + hybrid evaluator (VLM + SigLIP + OCR) that turns a product + brand + brief into a 3-asset multi-channel campaign — not a `prompt -> image` tool. It's a portfolio/resume project, so the architecture (provider-abstracted generation, async 202/poll contract, stored attempt history) is the interview story; keep it clean enough to narrate.

## Never do these
- Never let the UX become `prompt -> generate image`; always go product + brand + brief -> planned campaign.
- Never block a request on image generation or LLM calls — use `POST -> 202` + background task + client polling `GET`.
- Never mix LangGraph reasoning with provider execution code; route all FLUX/VLM HTTP through `ImageGenerationProvider` / `VisionEvaluator`.
- Never overwrite a `GenerationAttempt`; each `CreativeAsset` keeps up to 3 attempts, each with its own `Evaluation`.
- Never let a high `overall_score` pass an asset with `product_fidelity < 0.92` — use `evaluate_pass` / `PASS_THRESHOLDS` hard constraints.
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
- `README.md` — placeholder, not written
- `pyproject.toml` — backend deps + pytest config (asyncio_mode=auto)
- `docker-compose.yml` — local Postgres 16 (ciba/ciba_dev/ciba_db)
- `alembic.ini` + `alembic/env.py` — Alembic config; env.py reads `DATABASE_URL` from `app.config`
- `alembic/versions/19baeac8e423_initial_schema.py` — initial migration (works around circular FKs)
- `app/config.py` — env settings; generation vars default to mock/dev-safe
- `app/database.py` — async engine, `AsyncSessionLocal`, `get_db`, test-injectable `get_session_factory`
- `app/models.py` — frozen schema: User, Brand, Product, Campaign, CreativeConcept, CreativeAsset, GenerationAttempt, Evaluation
- `app/auth.py` — `get_current_user`: verifies NextAuth HS256 JWT, upserts `User` on `google_sub`
- `app/graph.py` — LangGraph (Director, Planner, gen/eval nodes) + provider interfaces/mocks + `PASS_THRESHOLDS`/`evaluate_pass` + Gemini Flash call
- `app/routes.py` — FastAPI contract: CRUD, campaign 202->poll, select-concept, assets poll, manual regenerate, background tasks
- `app/main.py` — FastAPI app, CORS, mounts router at `/api/v1`
- `tests/conftest.py` — ensures `ciba_test` DB exists
- `tests/test_smoke.py` — Checkpoint 1: brand/product CRUD
- `tests/test_auth_integration.py` — real JWT -> user upsert (no dependency overrides)
- `tests/test_e2e_checkpoint2.py` — full campaign flow with mocked generation/LLM
- `frontend/` — Next.js 16 + NextAuth v4 scaffolding; `app/api/auth/[...nextauth]/options.ts` uses custom HS256 JWS encode/decode (not NextAuth's default JWE) so FastAPI's python-jose can verify; `page.tsx`/`layout.tsx` are boilerplate
- `notebooks/flux-quality-test-working.ipynb` — standalone FLUX.2 Klein quality test (not integrated)

## Current state
> ⚠️ UPDATE ONLY THIS SECTION when a checkpoint lands.

Checkpoints 1 and 2 are done: Postgres + migration + JWT auth + brand/product CRUD (with smoke/auth tests), and the real Gemini Flash Director loop with mock image/eval providers running the full campaign -> concepts -> select -> 3 assets flow (e2e-tested). Checkpoint 3 is mostly implemented: `RemoteFluxKaggleProvider` now POSTs to a tunneled Kaggle endpoint (`KAGGLE_GATEWAY_URL`) and persists PNGs to `MEDIA_DIR`, and `_run_generation_loop` runs the full 3-attempt generate -> evaluate -> diagnose loop (covered by `tests/test_retry_loop.py`); still unverified against a live Kaggle worker. Checkpoint 4 (real VLM/SigLIP/OCR evaluation) and Checkpoint 5 (frontend UI — only NextAuth scaffolding exists) are not started.

## When in doubt
Prefer the boring Postgres-only solution over new infrastructure. Flag assumptions instead of guessing silently on anything touching the schema, auth, or retry logic.
