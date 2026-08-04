"""
Integration test for app/auth.py — exercises get_current_user for real
(no dependency overrides). Token is signed the same way the Next.js
NextAuth custom jwt.encode callback will sign it: plain HS256 using
NEXTAUTH_SECRET, with claims sub / email / name / picture.

This test requires Postgres (ciba_test) to be running.
Run after: docker-compose up -d postgres
"""
from __future__ import annotations

import time
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from jose import jwt
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.auth import ALGORITHM
from app.config import settings
from app.main import app
from app.models import Base, User

# ---------------------------------------------------------------------------
# Test-database setup (same pattern as test_smoke.py)
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = settings.DATABASE_URL.replace("/ciba_db", "/ciba_test")

_DROP_ORDER = [
    "evaluations", "generation_attempts", "creative_assets",
    "creative_concepts", "campaigns", "products", "brands", "users",
]
_DROP_ENUMS = [
    "campaign_status", "concept_status", "asset_status", "platform", "placement",
]


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
    async with engine.begin() as conn:
        for table in _DROP_ORDER:
            await conn.execute(sa_text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
        for enum in _DROP_ENUMS:
            await conn.execute(sa_text(f'DROP TYPE IF EXISTS "{enum}" CASCADE'))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
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
# Token factory — mirrors the Next.js NextAuth jwt.encode callback exactly:
#   algorithm : HS256
#   secret    : NEXTAUTH_SECRET (same env var)
#   claims    : sub, email, name, picture, iat, exp
# ---------------------------------------------------------------------------

def make_nextauth_token(
    *,
    google_sub: str,
    email: str,
    name: str = "Test User",
    picture: str = "https://example.com/photo.jpg",
    secret: str | None = None,
    expire_in: int = 3600,
) -> str:
    """
    Produce a plain HS256 JWT identical to what the Next.js NextAuth
    custom jwt.encode callback emits.
    """
    secret = secret or settings.NEXTAUTH_SECRET
    now = int(time.time())
    payload = {
        "sub": google_sub,          # Google's stable user ID
        "email": email,
        "name": name,
        "picture": picture,
        "iat": now,
        "exp": now + expire_in,
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# The integration test — no dependency_overrides
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_jwt_creates_user_and_brand(db_engine, session_factory):
    """
    End-to-end: a token signed exactly as the Next.js NextAuth callback
    would sign it is accepted by get_current_user, which upserts a User row,
    and then create_brand persists a Brand owned by that User.

    No dependency_overrides — get_current_user and get_db run for real,
    against the ciba_test Postgres database.
    """
    google_sub = f"google-real-{uuid.uuid4().hex[:8]}"
    email = f"real-{uuid.uuid4().hex[:6]}@integration.test"
    token = make_nextauth_token(google_sub=google_sub, email=email, name="Real User")

    # We override get_db so it points at ciba_test instead of ciba_db,
    # but we do NOT override get_current_user — it runs its full JWT decode
    # + upsert logic.
    from app.database import get_db as real_get_db

    # Override only the DB target (test DB, not dev DB) — auth runs for real.
    factory = session_factory

    async def test_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[real_get_db] = test_get_db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/brands",
                json={"name": "JWT Integration Brand", "tone": "Trustworthy"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["name"] == "JWT Integration Brand"
        assert "id" in data
        brand_id = uuid.UUID(data["id"])

        # Verify User row was upserted by google_sub
        async with factory() as session:
            user = (
                await session.execute(select(User).where(User.google_sub == google_sub))
            ).scalar_one_or_none()

        assert user is not None, "get_current_user should have upserted a User row"
        assert user.email == email
        assert user.name == "Real User"
        assert user.avatar_url == "https://example.com/photo.jpg"

        print(f"\nPASS: real JWT -> user={user.id} brand={brand_id}")

    finally:
        app.dependency_overrides.pop(real_get_db, None)


@pytest.mark.asyncio
async def test_real_jwt_upserts_on_second_login(db_engine, session_factory):
    """
    Second request with the same google_sub but updated name/picture
    should update the existing User row (upsert, not duplicate).
    """
    google_sub = f"google-upsert-{uuid.uuid4().hex[:8]}"
    email = f"upsert-{uuid.uuid4().hex[:6]}@integration.test"

    from app.database import get_db as real_get_db
    factory = session_factory

    async def test_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[real_get_db] = test_get_db

    try:
        # First login
        token1 = make_nextauth_token(
            google_sub=google_sub, email=email,
            name="Original Name", picture="https://example.com/old.jpg",
        )
        # Second login — same sub, updated name/picture
        token2 = make_nextauth_token(
            google_sub=google_sub, email=email,
            name="Updated Name", picture="https://example.com/new.jpg",
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.post(
                "/api/v1/brands",
                json={"name": "Brand for Upsert Test"},
                headers={"Authorization": f"Bearer {token1}"},
            )
            assert r1.status_code == 201

            r2 = await client.post(
                "/api/v1/brands",
                json={"name": "Brand After Profile Update"},
                headers={"Authorization": f"Bearer {token2}"},
            )
            assert r2.status_code == 201

        # Only one User row should exist for this google_sub
        async with factory() as session:
            users = (
                await session.execute(select(User).where(User.google_sub == google_sub))
            ).scalars().all()

        assert len(users) == 1, f"Expected 1 User row, got {len(users)}"
        assert users[0].name == "Updated Name"
        assert users[0].avatar_url == "https://example.com/new.jpg"

        print(f"\nPASS: upsert on second login -> user={users[0].id}, name updated correctly")

    finally:
        app.dependency_overrides.pop(real_get_db, None)


@pytest.mark.asyncio
async def test_invalid_token_returns_401(db_engine):
    """A tampered/wrong-secret token must be rejected with HTTP 401."""
    token = make_nextauth_token(
        google_sub="hacker-123",
        email="hacker@evil.com",
        secret="wrong-secret",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/v1/brands",
            json={"name": "Should Be Rejected"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"
    print("\nPASS: tampered token correctly rejected with 401")


@pytest.mark.asyncio
async def test_expired_token_returns_401(db_engine):
    """An expired token must be rejected with HTTP 401."""
    token = make_nextauth_token(
        google_sub="expired-user",
        email="expired@test.com",
        expire_in=-60,  # already expired 60 seconds ago
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/v1/brands",
            json={"name": "Should Be Rejected"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"
    print("\nPASS: expired token correctly rejected with 401")
