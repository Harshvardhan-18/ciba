# Agentic Creative Campaign Engine — Build Brief for Claude Code

## What this is

An agentic system that turns a product + brand context + campaign brief into
a full multi-channel ad campaign — not a `prompt -> image` generator. A
LangGraph "Creative Director" proposes campaign concepts; once the user
picks one, an "Asset Planner" turns it into per-channel image specs; a
generation provider (FLUX.2 Klein) renders them; a hybrid evaluator
(VLM + SigLIP + OCR) scores product fidelity and triggers self-correction
before an asset is approved.

This is a resume/portfolio project. Priorities, in order: (1) a working
end-to-end vertical slice, (2) clean architecture that's easy to narrate in
an interview, (3) breadth of infra. Do not let (3) delay (1).

## Non-negotiable architectural rules

1. **Never let the core UX become `prompt -> generate image`.** The user
   provides product + brand + brief; the system plans and produces a
   coherent campaign.
2. **Reference-conditioned generation, not text-to-image.** Every FLUX call
   takes the real product image(s) as input and must preserve product
   identity. This is the project's signature feature — treat product
   fidelity evaluation as more important than making pretty pictures.
3. **LangGraph does reasoning. Provider abstractions do execution.**
   LangGraph: creative direction, concept planning, evaluation diagnosis,
   corrective-prompt authoring. Providers: actual image generation, actual
   vision scoring. Never mix these — e.g. don't put HTTP calls to FLUX
   inside a LangGraph node's business logic without going through
   `ImageGenerationProvider`.
4. **Everything generation-related is asynchronous.** No endpoint blocks on
   image generation or LLM calls. Pattern: `POST -> 202 Accepted with a
   status field -> background task -> client polls GET`. This must hold
   even in V1 with no Kafka, because the frontend contract shouldn't change
   when Kafka is introduced later.
5. **Store every generation attempt, never overwrite.** Each `CreativeAsset`
   accumulates up to 3 `GenerationAttempt` rows (1 initial + 2 retries),
   each with its own `Evaluation`. This "system improved its output" history
   is a deliberate demo feature, not incidental logging.
6. **Evaluation uses hard constraints, not just a weighted average.** A
   creative with a beautiful score but `product_fidelity < 0.92` must fail
   regardless of overall_score. Never let one number average away a broken
   product.
7. **Provider-swappable by design.** `ImageGenerationProvider` and
   `VisionEvaluator` are abstract interfaces. Ship `Mock` implementations
   first and default to them in dev (`IMAGE_GENERATION_PROVIDER=mock`).
   Only call the real FLUX/VLM providers when explicitly configured, to
   avoid burning Kaggle GPU quota or API budget while iterating on
   unrelated code.
7a. **Two-tier generation strategy — build vs. showcase.** All development,
   testing, and the deployed reference implementation use
   `RemoteFluxKaggleProvider` (free, Kaggle T4). A separate
   `GeminiImageProvider` (paid) exists solely to record the final portfolio
   demo video at higher quality. Same interface, swapped via
   `IMAGE_GENERATION_PROVIDER=gemini`, zero application-layer changes.
   Never make Gemini the default provider, never call it in tests/CI, and
   treat it as something enabled briefly and turned back off. The
   interview narrative this supports: "generation is provider-agnostic;
   FLUX.2 Klein on free GPU is the real deployed system, and I can
   swap in a paid provider with a one-line config change when quality
   matters more than cost" — the code should make that literally true,
   not just a claim.
8. **Kaggle is disposable GPU compute, not a server.** Do not build the app
   around synchronously calling a Kaggle notebook endpoint. The intended
   shape (build later, after the vertical slice works): Kaggle notebook
   polls/consumes jobs from a small authenticated HTTPS gateway that the
   local backend exposes — Kaggle initiates the outbound connection, not
   the other way around.

## What is explicitly OUT of scope for V1 — do not build these yet

- Qdrant, embeddings, any vector retrieval (Brand context is just queried
  from Postgres and injected directly into the Director prompt)
- Kafka (background tasks / asyncio are sufficient for V1's job queue)
- Kubernetes, KEDA, Docker Compose beyond local Postgres
- S3/MinIO (use local disk for images in V1)
- OpenTelemetry/Prometheus/Grafana
- Channel selection UI, campaign kit presets, more than 3 asset types
- Billing, teams, multi-brand permissions, anything beyond Google OAuth

If you find yourself wanting to add one of these to solve a V1 problem,
stop and prefer the boring solution (e.g. a Postgres query, a Python
in-process loop) instead. These get added later, each justified by a
concrete bottleneck the vertical slice actually hits — that sequencing is
part of the project's story.

## Frozen V1 scope

- **3 fixed assets per campaign**: Instagram Feed (4:5, 1080x1350),
  Instagram Story (9:16, 1080x1920), Website Hero (16:9, 1920x1080). No
  channel picker — every campaign generates all three.
- **Director generates 2-3 concepts** (cheap LLM text/JSON output only).
  User selects ONE. Only the selected concept goes to image generation —
  never generate images for unselected concepts (Kaggle GPU is scarce).
- **Planner generates 3 AssetSpecs** from the selected concept, one per
  fixed placement, each with different composition (product position,
  framing, copy-safe area) even though they share the same Visual DNA —
  this cross-platform recomposition (not crop-and-resize) is the second
  headline feature.
- **Retry policy**: max 3 total generation attempts per asset (1 + 2
  retries). Below-threshold on attempt 3 -> `MANUAL_REVIEW`, not another
  retry. Infra failures (provider crashes) are a separate failure mode
  (`INFRA_FAILED`) from quality failures — do not count them against the
  3-attempt quality budget.
- **Auth**: NextAuth.js with Google provider on the frontend. FastAPI
  verifies the NextAuth-issued JWT and upserts a `User` row keyed on
  `google_sub`.
- **Generation provider**: FLUX.2 Klein 4B via Kaggle T4 for build/dev/deployed
  reference implementation. Paid Gemini image generation is swapped in
  (via `IMAGE_GENERATION_PROVIDER=gemini`) only for recording the final
  showcase demo video — see rule 7a.
- **VLM evaluator**: hosted API with a usable free tier (verify current
  options at implementation time rather than assuming a vendor — Gemini is
  a likely candidate but confirm current limits first).
- **Similarity**: local SigLIP. **Text verification**: local OCR.

## Repo / stack

- Backend: FastAPI + SQLAlchemy 2.0 (async) + PostgreSQL + Alembic + LangGraph
- Frontend: Next.js (App Router) + NextAuth.js
- Python package management: your choice (uv/poetry acceptable), but keep
  it simple — no monorepo tooling beyond what's needed
- No Docker required for V1 dev; a docker-compose for local Postgres is fine

## Existing frozen artifacts — read these first, do not redesign them

Three files already exist and represent the frozen schema/contract. Treat
them as the source of truth for the domain model, LangGraph shape, and API
contract. Extend and implement them; don't restructure without a strong
reason, and if you think one is wrong, flag it rather than silently
diverging:

- `app/models.py` — SQLAlchemy models: User, Brand, Product, Campaign,
  CreativeConcept, CreativeAsset, GenerationAttempt, Evaluation
- `app/graph.py` — LangGraph shape: `build_director_graph`,
  `build_planning_graph`, generation/evaluation loop nodes, plus the
  `ImageGenerationProvider` / `VisionEvaluator` abstract interfaces and
  `MockImageProvider` / `MockVisionEvaluator` implementations
- `app/routes.py` — FastAPI route contract with Pydantic schemas and the
  `202 -> poll` pattern; endpoint bodies are stubbed (`...`) and need
  implementing

## Build order

Work in this order. Get each checkpoint fully working (including a manual
end-to-end test) before moving to the next — don't parallelize across
checkpoints.

1. **Foundation**: Postgres running locally, Alembic migration from
   `models.py`, FastAPI app skeleton, NextAuth JWT verification dependency,
   DB session dependency. Smoke test: create a user, brand, product via API.
2. **Director loop, mocked generation**: Implement `create_campaign` +
   `generate_concepts_node` for real (actual LLM call, structured JSON
   output matching `CreativeConcept` schema) but keep image generation on
   `MockImageProvider` and evaluation on `MockVisionEvaluator`. Get a full
   campaign -> concepts -> select -> 3 mock "generated" assets loop working
   end to end through the API (Swagger is fine, no frontend yet).
3. **Real image generation**: Test FLUX.2 Klein quality on Kaggle
   standalone (notebook, not integrated) before wiring it into
   `RemoteFluxKaggleProvider`. If quality is insufficient, that's a
   generation-strategy problem to solve before more integration work, not
   after.
4. **Real evaluation**: Wire the real VLM provider + SigLIP + OCR into
   `VisionEvaluator`, implement the hybrid scoring and hard-constraint pass
   logic, confirm the retry/diagnosis loop actually improves scores across
   attempts on a real example.
5. **Frontend**: Next.js UI around the now-stable API — brand/product
   setup, campaign brief form, concept selection, asset gallery showing
   per-attempt history for at least one asset (this is the best demo
   screen, don't skip it).
6. **Stop.** Everything after this (Qdrant, Kafka, K8s, observability) is
   a separate, later phase — do not start it as part of this build.

## Working style

- Confirm your understanding of an ambiguous point by proposing a specific
  default and proceeding, rather than blocking on a question, unless the
  choice would be expensive to reverse (e.g. a schema change).
- Prefer editing/extending the three frozen files over rewriting them.
- Keep provider implementations behind their abstract interfaces at all
  times, even during quick prototyping — this is the one piece of
  "infrastructure for its own sake" that's worth keeping from day one,
  because it's cheap now and expensive to retrofit.
