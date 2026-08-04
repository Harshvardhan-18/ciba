"""
FastAPI application entrypoint.

Mounts the router from app/routes.py. Async lifespan context manager is used
(the recommended pattern in FastAPI 0.95+) so there's a clean place to add
startup/shutdown hooks later (e.g. warming LangGraph graphs, closing DB pools).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — nothing needed yet; Alembic handles migrations separately.
    yield
    # Shutdown — nothing needed yet.


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

app.include_router(router, prefix="/api/v1")
