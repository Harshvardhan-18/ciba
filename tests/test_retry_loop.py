"""
Regression tests for the full generate -> evaluate -> diagnose -> regenerate
loop in app/routes._run_generation_loop (Checkpoint 3).

Exercises the loop directly (no HTTP client, no real providers) with fake
providers/evaluators so it is fast and deterministic:
  1. 3 quality failures   -> MANUAL_REVIEW, corrective instructions accumulate
                             across attempts, every attempt is persisted.
  2. Fail-then-pass       -> APPROVED on attempt 2, approved_attempt_id set.
  3. Repeated infra crash -> INFRA_FAILED, and infra rows do NOT consume the
                             quality attempt budget (all keep attempt_number=1).

Run: docker-compose up -d postgres && pytest tests/test_retry_loop.py -v
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload
from sqlalchemy.pool import NullPool

from app.models import (
    Base, User, Brand, Product, Campaign, CampaignStatus,
    CreativeConcept, CreativeAsset, AssetStatus, GenerationAttempt, Evaluation,
    Platform, Placement,
)
from app.graph import VisionEvaluator, ImageGenerationProvider
from app import routes as routes_module

TEST_DATABASE_URL = "postgresql+asyncpg://ciba:ciba_dev@localhost:5432/ciba_test"

_DROP_ORDER = [
    "evaluations", "generation_attempts", "creative_assets",
    "creative_concepts", "campaigns", "products", "brands", "users",
]
_DROP_ENUMS = [
    "campaign_status", "concept_status", "asset_status", "platform", "placement",
]


def _failing_scores() -> dict:
    return {
        "vlm_product_score": 0.5, "siglip_similarity": 0.5, "ocr_text_score": 1.0,
        "product_fidelity": 0.5, "brand_consistency": 0.5, "composition_score": 0.5,
        "prompt_alignment": 0.5, "overall_score": 0.5,
        "critical_text_error": False, "passed": False, "failure_reason": "product_fidelity_below_threshold",
    }


def _passing_scores() -> dict:
    return {
        "vlm_product_score": 0.95, "siglip_similarity": 0.93, "ocr_text_score": 1.0,
        "product_fidelity": 0.94, "brand_consistency": 0.92, "composition_score": 0.91,
        "prompt_alignment": 0.92, "overall_score": 0.93,
        "critical_text_error": False, "passed": True, "failure_reason": None,
    }


class FakeImageProvider(ImageGenerationProvider):
    def __init__(self):
        self.calls = 0

    async def generate(self, product_images, reference_images, prompt, width, height) -> str:
        self.calls += 1
        return f"/local/mock/generated_{self.calls}.jpg"


class InfraProvider(ImageGenerationProvider):
    async def generate(self, product_images, reference_images, prompt, width, height) -> str:
        raise RuntimeError("simulated Kaggle outage")


class AlwaysFailEvaluator(VisionEvaluator):
    async def evaluate(self, original_product_images, generated_image, asset_spec, brand_context) -> dict:
        return _failing_scores()


class FailThenPassEvaluator(VisionEvaluator):
    def __init__(self):
        self.calls = 0

    async def evaluate(self, original_product_images, generated_image, asset_spec, brand_context) -> dict:
        self.calls += 1
        return _failing_scores() if self.calls == 1 else _passing_scores()


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


async def _seed_asset(session_factory) -> tuple[uuid.UUID, uuid.UUID]:
    """Create a minimal user -> brand -> product -> campaign -> concept -> asset chain."""
    async with session_factory() as db:
        user = User(
            google_sub=f"retry-{uuid.uuid4().hex[:8]}",
            email=f"retry-{uuid.uuid4().hex[:6]}@test",
            name="Retry Tester",
        )
        db.add(user)
        await db.flush()

        brand = Brand(owner_id=user.id, name="Retry Brand")
        db.add(brand)
        await db.flush()

        product = Product(brand_id=brand.id, name="Retry Widget")
        db.add(product)
        await db.flush()

        campaign = Campaign(
            brand_id=brand.id, product_id=product.id, owner_id=user.id,
            brief_text="Test brief", status=CampaignStatus.GENERATING_ASSETS,
        )
        db.add(campaign)
        await db.flush()

        concept = CreativeConcept(
            campaign_id=campaign.id, name="Retry Concept", description="desc",
            visual_dna={
                "palette": [], "lighting": "x", "environment": "x",
                "materials": [], "mood": [], "photography_style": "x",
            },
            copy={"headline": "H", "subcopy": None, "cta": None},
            rationale="r",
        )
        db.add(concept)
        await db.flush()

        asset = CreativeAsset(
            campaign_id=campaign.id, concept_id=concept.id,
            platform=Platform.INSTAGRAM, placement=Placement.IG_FEED,
            aspect_ratio="4:5", width=1080, height=1350,
            asset_spec={
                "generation_prompt": "Generate an ad.",
                "width": 1080, "height": 1350,
                "placement": "ig_feed", "platform": "instagram", "aspect_ratio": "4:5",
            },
            status=AssetStatus.PENDING,
        )
        db.add(asset)
        await db.flush()
        asset_id, campaign_id = asset.id, campaign.id
        await db.commit()
        return campaign_id, asset_id


async def _run_loop(monkeypatch, session_factory, asset_id, *, image_provider, evaluator):
    monkeypatch.setattr(routes_module, "get_image_provider", lambda: image_provider)
    monkeypatch.setattr(routes_module, "get_vision_evaluator", lambda: evaluator)
    await routes_module._run_generation_loop(
        asset_id=asset_id,
        product_images=[],
        brand_context={},
        session_factory=session_factory,
    )


@pytest.mark.asyncio
async def test_three_quality_failures_lands_manual_review(monkeypatch, session_factory):
    _, asset_id = await _seed_asset(session_factory)
    provider = FakeImageProvider()
    await _run_loop(
        monkeypatch, session_factory, asset_id,
        image_provider=provider, evaluator=AlwaysFailEvaluator(),
    )

    async with session_factory() as db:
        asset = (await db.execute(
            select(CreativeAsset).where(CreativeAsset.id == asset_id)
            .options(selectinload(CreativeAsset.attempts).selectinload(GenerationAttempt.evaluation))
        )).scalar_one()
        assert asset.status == AssetStatus.MANUAL_REVIEW
        assert provider.calls == 3
        attempts = sorted(asset.attempts, key=lambda a: a.attempt_number)
        assert [a.attempt_number for a in attempts] == [1, 2, 3]
        assert all(not a.infra_failed for a in attempts)
        assert all(a.evaluation is not None for a in attempts)
        assert all(a.evaluation.passed is False for a in attempts)
        # Corrective instruction should carry into attempts 2 and 3 only.
        assert attempts[0].corrective_instruction is None
        assert attempts[1].corrective_instruction is not None
        assert attempts[2].corrective_instruction is not None
        assert asset.approved_attempt_id is None
    print("\nPASS: 3 quality failures -> MANUAL_REVIEW, attempts preserved")


@pytest.mark.asyncio
async def test_approves_on_second_attempt(monkeypatch, session_factory):
    _, asset_id = await _seed_asset(session_factory)
    provider = FakeImageProvider()
    await _run_loop(
        monkeypatch, session_factory, asset_id,
        image_provider=provider, evaluator=FailThenPassEvaluator(),
    )

    async with session_factory() as db:
        asset = (await db.execute(
            select(CreativeAsset).where(CreativeAsset.id == asset_id)
            .options(selectinload(CreativeAsset.attempts).selectinload(GenerationAttempt.evaluation))
        )).scalar_one()
        assert asset.status == AssetStatus.APPROVED
        assert provider.calls == 2
        attempts = sorted(asset.attempts, key=lambda a: a.attempt_number)
        assert len(attempts) == 2
        assert attempts[0].evaluation.passed is False
        assert attempts[1].evaluation.passed is True
        assert attempts[1].corrective_instruction is not None
        assert asset.approved_attempt_id == attempts[1].id
    print("\nPASS: fail -> diagnose -> pass => APPROVED on attempt 2")


@pytest.mark.asyncio
async def test_infra_failures_do_not_consume_quality_budget(monkeypatch, session_factory):
    _, asset_id = await _seed_asset(session_factory)
    await _run_loop(
        monkeypatch, session_factory, asset_id,
        image_provider=InfraProvider(), evaluator=AlwaysFailEvaluator(),
    )

    async with session_factory() as db:
        asset = (await db.execute(
            select(CreativeAsset).where(CreativeAsset.id == asset_id)
            .options(selectinload(CreativeAsset.attempts).selectinload(GenerationAttempt.evaluation))
        )).scalar_one()
        assert asset.status == AssetStatus.INFRA_FAILED
        assert len(asset.attempts) == 3
        assert all(a.infra_failed for a in asset.attempts)
        # Infra rows don't advance the quality attempt counter — all are #1.
        assert {a.attempt_number for a in asset.attempts} == {1}
        assert all(a.evaluation is None for a in asset.attempts)
    print("\nPASS: infra crashes -> INFRA_FAILED, quality budget untouched")
