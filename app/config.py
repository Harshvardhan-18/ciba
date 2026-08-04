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

    # --- App ---
    DEBUG: bool = True
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]


settings = Settings()
