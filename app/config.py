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

    # --- App ---
    DEBUG: bool = True
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]


settings = Settings()
