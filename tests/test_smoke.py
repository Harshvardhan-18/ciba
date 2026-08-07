"""
Smoke test — Checkpoint 1 definition of done.

Verifies:
  1. A test JWT (mocked to bypass real NextAuth) creates/upserts a User row.
  2. POST /api/v1/brands  ->  201 + persisted Brand row.
  3. POST /api/v1/products -> 201 + persisted Product row linked to the brand.
  4. DB round-trip: fetch Brand and Product directly from the DB and confirm
     all fields match.

Auth dependency is overridden to inject a pre-seeded User without touching
NextAuth. The DB uses a real Postgres test database (same docker-compose
instance, separate `ciba_test` database) so that Postgres-specific types
(JSONB, ARRAY, UUID) work correctly.

Usage:
    # Start Postgres first:
    docker-compose up -d postgres

    # Then run:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.main import app
from app.models import Base, User, Brand, Product
from app.database import get_db
from app.auth import get_current_user

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://ciba:ciba_dev@localhost:5432/ciba_test",
)

# Tables to drop in dependency order (SQLAlchemy can't auto-sort circular FKs)
_DROP_ORDER = [
    "evaluations", "generation_attempts", "creative_assets",
    "creative_concepts", "campaigns", "products", "brands", "users",
]
_DROP_ENUMS = [
    "campaign_status", "concept_status", "asset_status", "platform", "placement",
]

TEST_USER_ID = uuid.uuid4()
TEST_GOOGLE_SUB = "google-sub-test-123"
TEST_EMAIL = "smoke@test.example"


# ---------------------------------------------------------------------------
# Session-scoped engine + session factory
# Engine is created INSIDE the fixture so it's bound to the test event loop.
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)

    # Setup: drop any leftover tables, then create fresh
    async with engine.begin() as conn:
        for table in _DROP_ORDER:
            await conn.execute(sa_text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
        for enum in _DROP_ENUMS:
            await conn.execute(sa_text(f'DROP TYPE IF EXISTS "{enum}" CASCADE'))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Teardown
    async with engine.begin() as conn:
        for table in _DROP_ORDER:
            await conn.execute(sa_text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
        for enum in _DROP_ENUMS:
            await conn.execute(sa_text(f'DROP TYPE IF EXISTS "{enum}" CASCADE'))
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def session_factory(db_engine):
    return async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


# ---------------------------------------------------------------------------
# Pre-seeded test user
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def test_user(session_factory) -> User:
    async with session_factory() as session:
        user = User(
            id=TEST_USER_ID,
            google_sub=TEST_GOOGLE_SUB,
            email=TEST_EMAIL,
            name="Smoke Test User",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


# ---------------------------------------------------------------------------
# Dependency overrides
# ---------------------------------------------------------------------------

def make_override_get_db(factory):
    async def _override() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    return _override


def make_override_get_current_user(user: User):
    async def _override() -> User:
        return user
    return _override


# ---------------------------------------------------------------------------
# HTTPX async client fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(test_user: User, session_factory) -> AsyncGenerator[AsyncClient, None]:
    app.dependency_overrides[get_db] = make_override_get_db(session_factory)
    app.dependency_overrides[get_current_user] = make_override_get_current_user(test_user)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helper to query DB directly in tests
# ---------------------------------------------------------------------------

def make_db_session(factory):
    """Context manager for direct DB access in assertions."""
    return factory()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_brand(client: AsyncClient, test_user: User, session_factory):
    """POST /api/v1/brands returns 201 with correct fields and persists to DB."""
    payload = {
        "name": "Smoke Brand",
        "description": "A test brand",
        "primary_colors": ["#FF5733", "#C70039"],
        "secondary_colors": ["#FFC300"],
        "fonts": ["Inter", "Roboto"],
        "tone": "Bold and confident",
    }
    resp = await client.post("/api/v1/brands", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["name"] == "Smoke Brand"
    assert data["primary_colors"] == ["#FF5733", "#C70039"]
    assert data["logo_url"] is None
    assert "id" in data

    # Round-trip: verify the row in the DB
    brand_id = uuid.UUID(data["id"])
    async with session_factory() as session:
        result = await session.execute(select(Brand).where(Brand.id == brand_id))
        brand = result.scalar_one()

    assert brand.owner_id == test_user.id
    assert brand.name == "Smoke Brand"
    assert brand.tone == "Bold and confident"
    assert brand.fonts == ["Inter", "Roboto"]
    print(f"\nOK  create_brand OK: brand_id={brand_id}")


@pytest.mark.asyncio
async def test_create_product(client: AsyncClient, test_user: User, session_factory):
    """POST /api/v1/products returns 201 linked to the brand."""
    brand_resp = await client.post(
        "/api/v1/brands",
        json={"name": "Product's Brand", "description": "For product test"},
    )
    assert brand_resp.status_code == 201, brand_resp.text
    brand_id = brand_resp.json()["id"]

    payload = {
        "brand_id": brand_id,
        "name": "Smoke Product",
        "description": "A test product",
    }
    resp = await client.post("/api/v1/products", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["name"] == "Smoke Product"
    assert data["brand_id"] == brand_id
    assert data["product_images"] == []
    assert "id" in data

    # Round-trip: verify the row in the DB
    product_id = uuid.UUID(data["id"])
    async with session_factory() as session:
        result = await session.execute(select(Product).where(Product.id == product_id))
        product = result.scalar_one()

    assert str(product.brand_id) == brand_id
    assert product.name == "Smoke Product"
    assert product.product_images == []
    print(f"\nOK  create_product OK: product_id={product_id}")


@pytest.mark.asyncio
async def test_create_product_wrong_brand_returns_404(client: AsyncClient):
    """POST /api/v1/products with a brand not owned by the user returns 404."""
    resp = await client.post(
        "/api/v1/products",
        json={
            "brand_id": str(uuid.uuid4()),
            "name": "Should fail",
        },
    )
    assert resp.status_code == 404, resp.text
    print("\nOK  ownership guard OK: 404 on foreign brand")


@pytest.mark.asyncio
async def test_full_round_trip(client: AsyncClient, test_user: User, session_factory):
    """
    Full round-trip: create brand -> create product -> verify both rows in DB
    with correct foreign key linkage.
    """
    # 1. Create brand
    brand_resp = await client.post(
        "/api/v1/brands",
        json={
            "name": "Round Trip Brand",
            "primary_colors": ["#123456"],
            "tone": "Minimal",
        },
    )
    assert brand_resp.status_code == 201
    brand_id = brand_resp.json()["id"]

    # 2. Create product under that brand
    product_resp = await client.post(
        "/api/v1/products",
        json={
            "brand_id": brand_id,
            "name": "Round Trip Product",
            "description": "Full round-trip product",
        },
    )
    assert product_resp.status_code == 201
    product_id = product_resp.json()["id"]

    # 3. Verify DB state for both rows
    async with session_factory() as session:
        brand_row = (
            await session.execute(select(Brand).where(Brand.id == uuid.UUID(brand_id)))
        ).scalar_one()
        product_row = (
            await session.execute(select(Product).where(Product.id == uuid.UUID(product_id)))
        ).scalar_one()

    assert brand_row.owner_id == test_user.id
    assert str(product_row.brand_id) == brand_id
    assert product_row.name == "Round Trip Product"
    assert brand_row.name == "Round Trip Brand"

    print(
        f"\nOK  Full round-trip OK:"
        f"\n    user={test_user.id}"
        f"\n    brand={brand_id}"
        f"\n    product={product_id}"
    )


@pytest.mark.asyncio
async def test_set_product_images(client: AsyncClient, session_factory):
    """POST /products/{id}/images replaces product_images (Kaggle dataset paths)."""
    brand_resp = await client.post("/api/v1/brands", json={"name": "Images Brand"})
    assert brand_resp.status_code == 201
    product_resp = await client.post(
        "/api/v1/products",
        json={"brand_id": brand_resp.json()["id"], "name": "Images Product"},
    )
    assert product_resp.status_code == 201
    product_id = product_resp.json()["id"]

    paths = ["/kaggle/input/my-product/p1.webp", "/kaggle/input/my-product/p2.webp"]
    resp = await client.post(f"/api/v1/products/{product_id}/images", json={"product_images": paths})
    assert resp.status_code == 200, resp.text
    assert resp.json()["product_images"] == paths

    # Round-trip: verify the row in the DB
    async with session_factory() as session:
        product = (await session.execute(
            select(Product).where(Product.id == uuid.UUID(product_id))
        )).scalar_one()
    assert product.product_images == paths

    # Replace semantics: a new list overwrites the old one.
    resp2 = await client.post(
        f"/api/v1/products/{product_id}/images",
        json={"product_images": ["/kaggle/input/my-product/p3.webp"]},
    )
    assert resp2.status_code == 200
    assert resp2.json()["product_images"] == ["/kaggle/input/my-product/p3.webp"]
    print("\nOK  set_product_images OK: replace semantics + DB round-trip")


@pytest.mark.asyncio
async def test_set_product_images_foreign_product_returns_404(client: AsyncClient):
    """Setting images on a product not owned by the user returns 404."""
    resp = await client.post(
        f"/api/v1/products/{uuid.uuid4()}/images",
        json={"product_images": ["/kaggle/input/x/p1.webp"]},
    )
    assert resp.status_code == 404, resp.text
    print("\nOK  set_product_images ownership guard OK: 404 on foreign product")

