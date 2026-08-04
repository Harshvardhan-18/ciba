"""
End-to-end test for Checkpoint 2 — Director loop with mocked generation.

Flow:
  1. Create user / brand / product (reuses Checkpoint 1 auth fixture pattern)
  2. POST /campaigns                  -> 202 (status=generating_concepts)
     [background: Director LangGraph -> concepts persisted]
  3. GET /campaigns/{id}              -> concepts_ready
  4. GET /campaigns/{id}/concepts     -> 2-3 ConceptOut rows with valid schema
  5. POST /campaigns/{id}/select-concept -> 202 (status=generating_assets)
     [background: Planner + mock generation loop]
  6. GET /campaigns/{id}/assets       -> 3 assets, all approved
     each with 1 GenerationAttempt + 1 Evaluation

BackgroundTasks behaviour with httpx ASGITransport:
  httpx's ASGITransport runs background tasks synchronously as part of the
  ASGI send() call, so by the time client.post() returns the background task
  has already completed. This means no polling loop is needed in tests —
  we can assert the final state immediately after each POST.

IMAGE_GENERATION_PROVIDER is forced to "mock" via env var so no real
FLUX/VLM credentials are required. LLM_PROVIDER is forced to "mock_llm"
which uses a canned concept response injected by monkeypatching
_call_gemini — no GEMINI_API_KEY required.

Run:
    docker-compose up -d postgres
    pytest tests/test_e2e_checkpoint2.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.main import app
from app.models import (
    Base, User, Brand, Product, Campaign, CampaignStatus,
    CreativeConcept, CreativeAsset, AssetStatus, GenerationAttempt, Evaluation,
)
from app.database import get_db
from app.auth import get_current_user

# Force mock providers — no real credentials needed
os.environ["IMAGE_GENERATION_PROVIDER"] = "mock"
os.environ["LLM_PROVIDER"] = "gemini"  # will be monkeypatched below

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://ciba:ciba_dev@localhost:5432/ciba_test",
)

_DROP_ORDER = [
    "evaluations", "generation_attempts", "creative_assets",
    "creative_concepts", "campaigns", "products", "brands", "users",
]
_DROP_ENUMS = [
    "campaign_status", "concept_status", "asset_status", "platform", "placement",
]

# ---------------------------------------------------------------------------
# Canned LLM response — replaces Gemini API call in tests
# ---------------------------------------------------------------------------

MOCK_CONCEPTS = [
    {
        "name": "Urban Pulse",
        "description": "Bold street-level photography capturing the product in motion.",
        "visual_dna": {
            "palette": ["#1A1A2E", "#E94560"],
            "lighting": "dramatic side lighting",
            "environment": "urban rooftop at dusk",
            "materials": ["concrete", "steel"],
            "mood": ["bold", "energetic"],
            "photography_style": "editorial",
        },
        "ad_copy": {
            "headline": "Own the Night",
            "subcopy": "For those who never stop.",
            "cta": "Shop Now",
        },
        "rationale": "Targets urban professionals who value performance aesthetics.",
    },
    {
        "name": "Quiet Luxury",
        "description": "Minimalist studio composition highlighting material quality.",
        "visual_dna": {
            "palette": ["#F5F0E8", "#2C2C2C"],
            "lighting": "soft diffused natural light",
            "environment": "minimal white studio",
            "materials": ["linen", "marble"],
            "mood": ["refined", "calm"],
            "photography_style": "clean product photography",
        },
        "ad_copy": {
            "headline": "Crafted to Last",
            "subcopy": None,
            "cta": "Explore",
        },
        "rationale": "Appeals to consumers who prioritize quality over trend.",
    },
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def db_engine():
    # Use a regular pool (not NullPool) so background tasks can open new
    # connections without the asyncpg transport becoming invalid.
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=False,   # skip pre-ping; NullPool+NullTransport causes issues
        pool_size=5,
        max_overflow=2,
    )
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


TEST_USER_ID = uuid.uuid4()
TEST_GOOGLE_SUB = f"google-e2e-{uuid.uuid4().hex[:8]}"
TEST_EMAIL = f"e2e-{uuid.uuid4().hex[:6]}@checkpoint2.test"


@pytest_asyncio.fixture(scope="session")
async def test_user(session_factory) -> User:
    async with session_factory() as session:
        user = User(
            id=TEST_USER_ID,
            google_sub=TEST_GOOGLE_SUB,
            email=TEST_EMAIL,
            name="Checkpoint2 User",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


# ---------------------------------------------------------------------------
# HTTP client fixture with DB + auth overrides
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(test_user: User, session_factory) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def override_get_current_user() -> User:
        return test_user

    import app.database as _db_module
    import app.routes as _routes_module
    original_factory = _db_module.get_session_factory

    # Patch get_session_factory on the routes module (which holds a local
    # reference to the imported function) so background tasks use the test DB.
    _db_module.get_session_factory = lambda: session_factory
    _routes_module.get_session_factory = lambda: session_factory

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()
    _db_module.get_session_factory = original_factory
    _routes_module.get_session_factory = original_factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for_campaign_status(
    client: AsyncClient,
    campaign_id: str,
    expected_status: str,
    max_polls: int = 20,
    poll_interval: float = 0.15,
) -> dict:
    """
    Poll GET /campaigns/{id} until status matches expected_status.
    With ASGITransport, background tasks run inline so this usually
    resolves on the first poll — the loop is a safety net.
    """
    for _ in range(max_polls):
        r = await client.get(f"/api/v1/campaigns/{campaign_id}")
        assert r.status_code == 200, r.text
        data = r.json()
        if data["status"] == expected_status:
            return data
        await asyncio.sleep(poll_interval)
    raise AssertionError(
        f"Campaign {campaign_id} never reached status={expected_status!r}. "
        f"Last status: {data['status']!r}"
    )


# ---------------------------------------------------------------------------
# The E2E test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_campaign_flow(client: AsyncClient, test_user: User, session_factory):
    """
    Full Checkpoint 2 flow:
      brand -> product -> campaign -> concepts_ready ->
      select concept -> generating_assets -> all assets approved
    """

    # ------------------------------------------------------------------
    # Step 1: Create brand and product (reuse Checkpoint 1 infra)
    # ------------------------------------------------------------------
    brand_resp = await client.post(
        "/api/v1/brands",
        json={
            "name": "Checkpoint2 Brand",
            "description": "A bold lifestyle brand",
            "primary_colors": ["#E94560", "#1A1A2E"],
            "fonts": ["Inter"],
            "tone": "Bold and aspirational",
        },
    )
    assert brand_resp.status_code == 201, brand_resp.text
    brand_id = brand_resp.json()["id"]

    product_resp = await client.post(
        "/api/v1/products",
        json={
            "brand_id": brand_id,
            "name": "Signature Sneaker",
            "description": "Limited edition premium sneaker with carbon sole",
        },
    )
    assert product_resp.status_code == 201, product_resp.text
    product_id = product_resp.json()["id"]

    print(f"\n[E2E] brand={brand_id} product={product_id}")

    # ------------------------------------------------------------------
    # Step 2: Create campaign — patch the LLM call to avoid API key
    # ------------------------------------------------------------------
    import json as _json

    async def mock_call_gemini(user_prompt: str) -> list[dict]:
        return MOCK_CONCEPTS

    with patch("app.graph._call_gemini", side_effect=mock_call_gemini):
        campaign_resp = await client.post(
            "/api/v1/campaigns",
            json={
                "brand_id": brand_id,
                "product_id": product_id,
                "brief_text": "Launch campaign for new sneaker targeting urban millennials",
                "target_audience": "Urban millennials 25-35",
            },
        )
        assert campaign_resp.status_code == 202, campaign_resp.text
        campaign_data = campaign_resp.json()
        campaign_id = campaign_data["id"]

        print(f"[E2E] campaign={campaign_id} initial status={campaign_data['status']}")

        # With ASGITransport, background tasks run inline, but let's poll
        # to be safe and confirm the status transitioned
        final_campaign = await _wait_for_campaign_status(
            client, campaign_id, "concepts_ready"
        )

    assert final_campaign["status"] == "concepts_ready", final_campaign
    print(f"[E2E] Campaign reached concepts_ready")

    # ------------------------------------------------------------------
    # Step 3: List concepts — validate schema
    # ------------------------------------------------------------------
    concepts_resp = await client.get(f"/api/v1/campaigns/{campaign_id}/concepts")
    assert concepts_resp.status_code == 200, concepts_resp.text
    concepts = concepts_resp.json()

    assert 2 <= len(concepts) <= 3, f"Expected 2-3 concepts, got {len(concepts)}"

    for c in concepts:
        assert "id" in c
        assert "name" in c and isinstance(c["name"], str)
        assert "description" in c and isinstance(c["description"], str)
        assert "visual_dna" in c and isinstance(c["visual_dna"], dict)
        assert "ad_copy" in c and isinstance(c["ad_copy"], dict), \
            f"ad_copy field missing or wrong type: {c.get('ad_copy')}"
        assert "headline" in c["ad_copy"], f"ad_copy missing headline: {c['ad_copy']}"
        assert "rationale" in c and isinstance(c["rationale"], str)
        assert "status" in c and c["status"] == "proposed"
        # Validate visual_dna required keys
        dna = c["visual_dna"]
        for key in ("palette", "lighting", "environment", "materials", "mood", "photography_style"):
            assert key in dna, f"visual_dna missing {key!r}: {dna}"

    print(f"[E2E] {len(concepts)} concepts validated:")
    for c in concepts:
        print(f"  - {c['name']}: {c['ad_copy']['headline']}")

    # ------------------------------------------------------------------
    # Step 4: Select first concept -> triggers Planner + generation loop
    # ------------------------------------------------------------------
    selected_concept_id = concepts[0]["id"]

    select_resp = await client.post(
        f"/api/v1/campaigns/{campaign_id}/select-concept",
        json={"concept_id": selected_concept_id},
    )
    assert select_resp.status_code == 202, select_resp.text
    assert select_resp.json()["status"] == "generating_assets"
    print(f"[E2E] Selected concept={selected_concept_id}")

    # ------------------------------------------------------------------
    # Step 5: Wait for assets to be generated (mock -> instant)
    # ------------------------------------------------------------------
    await _wait_for_campaign_status(client, campaign_id, "complete")
    print(f"[E2E] Campaign reached complete")

    # ------------------------------------------------------------------
    # Step 6: List assets — validate all 3 are approved
    # ------------------------------------------------------------------
    assets_resp = await client.get(f"/api/v1/campaigns/{campaign_id}/assets")
    assert assets_resp.status_code == 200, assets_resp.text
    assets = assets_resp.json()

    assert len(assets) == 3, f"Expected 3 assets, got {len(assets)}"

    expected_placements = {"ig_feed", "ig_story", "website_hero"}
    actual_placements = {a["placement"] for a in assets}
    assert actual_placements == expected_placements, \
        f"Wrong placements: {actual_placements}"

    for asset in assets:
        assert asset["status"] == "approved", \
            f"Asset {asset['placement']} has status={asset['status']!r}, expected approved"

        assert len(asset["attempts"]) == 1, \
            f"Asset {asset['placement']} has {len(asset['attempts'])} attempts, expected 1"

        attempt = asset["attempts"][0]
        assert attempt["attempt_number"] == 1
        assert not attempt["infra_failed"]
        assert attempt["image_url"] is not None

        evaluation = attempt["evaluation"]
        assert evaluation is not None, f"Asset {asset['placement']} missing evaluation"
        assert evaluation["passed"] is True
        assert evaluation["overall_score"] > 0
        assert evaluation["product_fidelity"] > 0
        assert evaluation["failure_reason"] is None

        print(
            f"  - {asset['placement']}: status={asset['status']}"
            f" score={evaluation['overall_score']:.2f}"
        )

    # ------------------------------------------------------------------
    # Step 7: DB-layer verification
    # ------------------------------------------------------------------
    async with session_factory() as session:
        # Verify campaign state
        campaign_row = (await session.execute(
            select(Campaign).where(Campaign.id == uuid.UUID(campaign_id))
        )).scalar_one()
        assert campaign_row.status == CampaignStatus.COMPLETE
        assert campaign_row.selected_concept_id == uuid.UUID(selected_concept_id)

        # Verify concepts in DB
        concept_rows = (await session.execute(
            select(CreativeConcept).where(CreativeConcept.campaign_id == uuid.UUID(campaign_id))
        )).scalars().all()
        assert len(concept_rows) == len(concepts)
        selected_rows = [c for c in concept_rows if str(c.id) == selected_concept_id]
        assert len(selected_rows) == 1
        assert selected_rows[0].status.value == "selected"
        rejected_rows = [c for c in concept_rows if str(c.id) != selected_concept_id]
        assert all(c.status.value == "rejected" for c in rejected_rows)

        # Verify asset -> attempt -> evaluation chain
        asset_rows = (await session.execute(
            select(CreativeAsset)
            .where(CreativeAsset.campaign_id == uuid.UUID(campaign_id))
            .options(selectinload(CreativeAsset.attempts).selectinload(GenerationAttempt.evaluation))
        )).scalars().all()
        assert len(asset_rows) == 3
        for asset_row in asset_rows:
            assert asset_row.status == AssetStatus.APPROVED
            assert asset_row.approved_attempt_id is not None
            assert len(asset_row.attempts) == 1
            attempt_row = asset_row.attempts[0]
            assert attempt_row.evaluation is not None
            assert attempt_row.evaluation.passed is True

            # Verify asset_spec has a non-null generation_prompt
            assert asset_row.asset_spec.get("generation_prompt"), \
                f"asset_spec.generation_prompt is empty for {asset_row.placement}"

    print("\nPASS: All DB assertions passed")
    print(f"  campaign={campaign_id} -> COMPLETE")
    print(f"  {len(concept_rows)} concepts ({sum(1 for c in concept_rows if c.status.value=='selected')} selected)")
    print(f"  {len(asset_rows)} assets, all APPROVED, each with 1 attempt + 1 evaluation")


@pytest.mark.asyncio
async def test_concept_out_uses_ad_copy_not_copy(client: AsyncClient, session_factory):
    """
    Regression: ConceptOut must use field name `ad_copy`, not `copy`.
    The old `copy` field shadowed Pydantic's BaseModel.model_copy and
    caused a UserWarning. Confirm the API response has `ad_copy` and
    NOT `copy` at the top level.
    """
    # Create minimal brand + product + campaign
    br = await client.post("/api/v1/brands", json={"name": "AdCopy Test Brand"})
    assert br.status_code == 201
    pr = await client.post("/api/v1/products", json={"brand_id": br.json()["id"], "name": "Widget"})
    assert pr.status_code == 201

    with patch("app.graph._call_gemini", side_effect=lambda _: asyncio.coroutine(lambda: MOCK_CONCEPTS)()):
        # Use AsyncMock for clean coroutine patching
        pass

    async def mock_gemini(_):
        return MOCK_CONCEPTS

    with patch("app.graph._call_gemini", side_effect=mock_gemini):
        cr = await client.post(
            "/api/v1/campaigns",
            json={
                "brand_id": br.json()["id"],
                "product_id": pr.json()["id"],
                "brief_text": "Test ad_copy field naming",
            },
        )
        assert cr.status_code == 202
        campaign_id = cr.json()["id"]
        await _wait_for_campaign_status(client, campaign_id, "concepts_ready")

    concepts_resp = await client.get(f"/api/v1/campaigns/{campaign_id}/concepts")
    assert concepts_resp.status_code == 200
    concepts = concepts_resp.json()
    assert len(concepts) > 0

    for concept in concepts:
        assert "ad_copy" in concept, f"Expected 'ad_copy' key, got keys: {list(concept.keys())}"
        assert "copy" not in concept, f"Old 'copy' key still present — rename not applied"
        assert "headline" in concept["ad_copy"]

    print("\nPASS: ConceptOut uses ad_copy field (not copy)")
