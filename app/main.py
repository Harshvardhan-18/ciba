"""
FastAPI application entrypoint.

Mounts the router from app/routes.py. Async lifespan context manager is used
(the recommended pattern in FastAPI 0.95+) so there's a clean place to add
startup/shutdown hooks later (e.g. warming LangGraph graphs, closing DB pools).
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.routes import recover_stuck_generation

    # Startup — resume any generation interrupted by a restart/reload.
    try:
        await recover_stuck_generation()
    except Exception:
        logger.exception("lifespan: recovery sweep failed (non-fatal)")

    # Self-healing loop: re-schedule campaigns/assets stuck mid-generation
    # (e.g. a uvicorn --reload killed the tasks) every 20s, so a run never
    # stays stuck until the next manual restart. Dedup is handled by the
    # in-process guard sets in routes.py.
    stop = asyncio.Event()

    async def recovery_loop() -> None:
        while not stop.is_set():
            try:
                await recover_stuck_generation()
            except Exception:
                logger.exception("recovery loop iteration failed (non-fatal)")
            try:
                await asyncio.wait_for(stop.wait(), timeout=20)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(recovery_loop())
    try:
        yield
    finally:
        stop.set()
        task.cancel()


app = FastAPI(
    title="Agentic Creative Campaign Engine",
    version="0.1.0",
    description="Turn a product + brand + brief into a full multi-channel ad campaign.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve locally-generated images (V1 local disk, no S3) so the frontend asset
# gallery can render per-attempt images at /media/<filename>.
os.makedirs(settings.MEDIA_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.MEDIA_DIR), name="media")

app.include_router(router, prefix="/api/v1")
