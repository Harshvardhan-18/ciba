"""
Settings loaded from environment variables (or a .env file via pydantic-settings).
All generation-related env vars default to safe/mock values so the app starts
without any external credentials during dev.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://ciba:ciba_dev@localhost:5432/ciba_db"

    # --- Auth ---
    # Secret used by NextAuth.js to sign JWTs (NEXTAUTH_SECRET env var on the frontend).
    # Must be set in production; a dummy value is provided so the app starts in tests.
    NEXTAUTH_SECRET: str = "dev-secret-change-me"

    # --- Generation provider ---
    # "mock" | "flux2_klein_kaggle" | "gemini"
    # Defaults to "mock" so no GPU/API quota is consumed during dev/test.
    IMAGE_GENERATION_PROVIDER: str = "mock"

    # --- Kaggle (RemoteFluxKaggleProvider) ---
    # Base URL of the tunneled Kaggle HTTPS endpoint that serves FLUX.2 Klein
    # (e.g. an ngrok/cloudflared URL). Empty by default so the provider can
    # never fire accidentally; set it only when a worker is actually running.
    KAGGLE_GATEWAY_URL: str = "https://a758-35-185-94-132.ngrok-free.app"
    # Optional shared secret sent as `Authorization: Bearer <key>` on each request.
    KAGGLE_API_KEY: str = ""
    # FLUX on a T4 is slow (roughly 1-2 min per image at production resolution).
    KAGGLE_REQUEST_TIMEOUT_SECONDS: float = 300.0

    # --- Image storage (V1: local disk, no S3) ---
    MEDIA_DIR: str = "media/generated"

    # --- Vision evaluator (Checkpoint 4) ---
    # "mock" | "hybrid". "hybrid" runs the real evaluation pipeline.
    # Defaults to "mock" so dev/tests don't need heavy deps or API quota.
    VISION_EVALUATOR_PROVIDER: str = "mock"

    # Which backend runs the hybrid evaluation when VISION_EVALUATOR_PROVIDER=hybrid.
    # "kaggle" (default, free) — VLM + SigLIP + OCR all run on the Kaggle eval
    #   worker (notebooks/kaggle_eval_worker.py); product images come from the
    #   Kaggle dataset paths. Requires KAGGLE_EVAL_GATEWAY_URL.
    # "gemini" — VLM via the Gemini API, SigLIP + OCR local on the backend;
    #   requires GEMINI_API_KEY and product images resolvable on the backend.
    # Swap later by changing ONLY this value.
    VLM_PROVIDER: str = "kaggle"

    # Tunnel URL of the Kaggle eval worker (separate notebook/session + T4 from
    # the generation worker). Empty by default so kaggle mode fails loudly
    # rather than firing at nothing.
    KAGGLE_EVAL_GATEWAY_URL: str = ""

    # Local SigLIP model used for product-fidelity similarity scoring
    # (gemini VLM mode only — kaggle mode runs SigLIP on the eval worker).
    SIGLIP_MODEL_ID: str = "google/siglip-so400m-patch14-384"

    # OCR: when overlay text IS detected, average confidence below this makes it
    # a critical_text_error (garbled copy must fail the asset).
    OCR_MIN_CONFIDENCE: float = 0.5

    # --- App ---
    DEBUG: bool = True
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]


settings = Settings()
