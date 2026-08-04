"""
API contract for the V1 vertical slice.

Rule from the frozen spec: nothing waits on generation synchronously.
Every state transition is create -> 202 Accepted -> background task ->
poll. This is what lets Kaggle/Kafka slot in later without changing the
frontend's interaction model.

    POST /campaigns                          202, status=generating_concepts
    GET  /campaigns/{id}                     poll -> concepts_ready
    POST /campaigns/{id}/select-concept      202, status=generating_assets
    GET  /campaigns/{id}/assets              poll -> approved | manual_review

ad_copy vs. copy
----------------
The DB column on CreativeConcept is named `copy` (frozen schema). The
Pydantic-facing field and graph-layer key is `ad_copy` to avoid shadowing
Python's built-in `copy` and Pydantic's `BaseModel.model_copy`. The route
layer translates between them when reading from / writing to the DB.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    Brand, Campaign, CampaignStatus, ConceptStatus, AssetStatus,
    CreativeAsset, CreativeConcept, Evaluation, GenerationAttempt,
    Platform, Placement, Product, User,
)
from app.auth import get_current_user
from app.database import get_db, get_session_factory
from app.graph import (
    build_director_graph, build_planning_graph,
    DirectorState, PlannerState, GenerationState,
    MockImageProvider, MockVisionEvaluator,
    ImageGenerationProvider, VisionEvaluator,
    generate_image_node, evaluate_node, diagnose_failure_node,
    FIXED_PLACEMENTS,
)

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3  # 1 initial + 2 regenerations, per frozen retry policy


# ---------------------------------------------------------------------------
# Provider factory — controlled by IMAGE_GENERATION_PROVIDER env var
# ---------------------------------------------------------------------------

def get_image_provider() -> ImageGenerationProvider:
    provider = os.environ.get("IMAGE_GENERATION_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockImageProvider()
    raise ValueError(f"Unknown IMAGE_GENERATION_PROVIDER: {provider!r}")


def get_vision_evaluator() -> VisionEvaluator:
    provider = os.environ.get("IMAGE_GENERATION_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockVisionEvaluator()
    raise ValueError(f"Unknown IMAGE_GENERATION_PROVIDER: {provider!r}")


# ---------------------------------------------------------------------------
# Brand / Product — plain CRUD, no vector store in V1
# ---------------------------------------------------------------------------

class BrandCreate(BaseModel):
    name: str
    description: str | None = None
    primary_colors: list[str] = Field(default_factory=list)
    secondary_colors: list[str] = Field(default_factory=list)
    fonts: list[str] = Field(default_factory=list)
    tone: str | None = None
    # logo uploaded via multipart on a separate endpoint: POST /brands/{id}/logo


class BrandOut(BrandCreate):
    id: uuid.UUID
    logo_url: str | None

    model_config = {"from_attributes": True}


@router.post("/brands", response_model=BrandOut, status_code=201)
async def create_brand(
    payload: BrandCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    brand = Brand(
        owner_id=user.id,
        name=payload.name,
        description=payload.description,
        primary_colors=payload.primary_colors,
        secondary_colors=payload.secondary_colors,
        fonts=payload.fonts,
        tone=payload.tone,
    )
    db.add(brand)
    await db.flush()
    return brand


class ProductCreate(BaseModel):
    brand_id: uuid.UUID
    name: str
    description: str | None = None
    # product_images uploaded via multipart on POST /products/{id}/images


class ProductOut(ProductCreate):
    id: uuid.UUID
    product_images: list[str]

    model_config = {"from_attributes": True}


@router.post("/products", response_model=ProductOut, status_code=201)
async def create_product(
    payload: ProductCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify the brand exists and belongs to the authenticated user.
    result = await db.execute(
        select(Brand).where(Brand.id == payload.brand_id, Brand.owner_id == user.id)
    )
    brand = result.scalar_one_or_none()
    if brand is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brand not found or not owned by current user.",
        )

    product = Product(
        brand_id=payload.brand_id,
        name=payload.name,
        description=payload.description,
        product_images=[],
    )
    db.add(product)
    await db.flush()
    return product


# ---------------------------------------------------------------------------
# Campaign creation -> Director runs in background -> concepts
# ---------------------------------------------------------------------------

class CampaignCreate(BaseModel):
    brand_id: uuid.UUID
    product_id: uuid.UUID
    brief_text: str
    target_audience: str | None = None


class CampaignOut(BaseModel):
    id: uuid.UUID
    status: CampaignStatus
    brief_text: str
    selected_concept_id: uuid.UUID | None

    model_config = {"from_attributes": True}


@router.post("/campaigns", response_model=CampaignOut, status_code=202)
async def create_campaign(
    payload: CampaignCreate,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Creates the Campaign row (status=generating_concepts) and schedules the
    Director graph as a background task. Client polls GET /campaigns/{id}
    until status flips to concepts_ready.
    """
    # Verify brand ownership
    brand = (await db.execute(
        select(Brand).where(Brand.id == payload.brand_id, Brand.owner_id == user.id)
    )).scalar_one_or_none()
    if brand is None:
        raise HTTPException(status_code=404, detail="Brand not found or not owned by current user.")

    # Verify product belongs to the brand
    product = (await db.execute(
        select(Product).where(Product.id == payload.product_id, Product.brand_id == payload.brand_id)
    )).scalar_one_or_none()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found or not linked to this brand.")

    campaign = Campaign(
        brand_id=payload.brand_id,
        product_id=payload.product_id,
        owner_id=user.id,
        brief_text=payload.brief_text,
        target_audience=payload.target_audience,
        status=CampaignStatus.GENERATING_CONCEPTS,
    )
    db.add(campaign)
    await db.flush()
    campaign_id = campaign.id

    # Snapshot brand/product context for the background task (can't pass DB objects)
    brand_context = {
        "name": brand.name,
        "description": brand.description,
        "primary_colors": brand.primary_colors,
        "secondary_colors": brand.secondary_colors,
        "fonts": brand.fonts,
        "tone": brand.tone,
    }
    product_context = {
        "name": product.name,
        "description": product.description,
        "image_urls": product.product_images,
    }

    # Commit explicitly so the campaign row is visible to the background task's
    # own DB session (background tasks run in a separate session).
    await db.commit()

    background.add_task(
        _run_director,
        campaign_id=campaign_id,
        brand_context=brand_context,
        product_context=product_context,
        brief_text=payload.brief_text,
        target_audience=payload.target_audience,
        session_factory=get_session_factory(),
    )

    return campaign


class ConceptOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    visual_dna: dict
    ad_copy: dict        # renamed from `copy` to avoid Pydantic BaseModel shadow warning
    rationale: str
    status: str

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_concept(cls, concept: CreativeConcept) -> "ConceptOut":
        """Maps DB column `copy` to Pydantic field `ad_copy`."""
        return cls(
            id=concept.id,
            name=concept.name,
            description=concept.description,
            visual_dna=concept.visual_dna,
            ad_copy=concept.copy,   # DB column `copy` -> Pydantic field `ad_copy`
            rationale=concept.rationale,
            status=concept.status.value,
        )


@router.get("/campaigns/{campaign_id}", response_model=CampaignOut)
async def get_campaign(
    campaign_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    campaign = (await db.execute(
        select(Campaign).where(Campaign.id == campaign_id, Campaign.owner_id == user.id)
    )).scalar_one_or_none()
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return campaign


@router.get("/campaigns/{campaign_id}/concepts", response_model=list[ConceptOut])
async def list_concepts(
    campaign_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Populated once campaign.status == concepts_ready."""
    campaign = (await db.execute(
        select(Campaign).where(Campaign.id == campaign_id, Campaign.owner_id == user.id)
    )).scalar_one_or_none()
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")

    concepts = (await db.execute(
        select(CreativeConcept).where(CreativeConcept.campaign_id == campaign_id)
    )).scalars().all()

    return [ConceptOut.from_orm_concept(c) for c in concepts]


# ---------------------------------------------------------------------------
# Concept selection -> Planner + generation loop run in background
# ---------------------------------------------------------------------------

class SelectConceptRequest(BaseModel):
    concept_id: uuid.UUID


@router.post("/campaigns/{campaign_id}/select-concept", response_model=CampaignOut, status_code=202)
async def select_concept(
    campaign_id: uuid.UUID,
    payload: SelectConceptRequest,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Marks the chosen CreativeConcept selected, the rest rejected. Creates
    the 3 fixed CreativeAsset rows (pending), runs the Planner graph to
    populate their asset_spec, then schedules generation for each asset.
    Sets campaign.status = generating_assets.
    """
    campaign = (await db.execute(
        select(Campaign).where(Campaign.id == campaign_id, Campaign.owner_id == user.id)
    )).scalar_one_or_none()
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")

    if campaign.status != CampaignStatus.CONCEPTS_READY:
        raise HTTPException(
            status_code=409,
            detail=f"Campaign is not in concepts_ready state (current: {campaign.status.value}).",
        )

    # Verify concept belongs to this campaign
    concept = (await db.execute(
        select(CreativeConcept).where(
            CreativeConcept.id == payload.concept_id,
            CreativeConcept.campaign_id == campaign_id,
        )
    )).scalar_one_or_none()
    if concept is None:
        raise HTTPException(status_code=404, detail="Concept not found in this campaign.")

    # Mark selected / rejected
    all_concepts = (await db.execute(
        select(CreativeConcept).where(CreativeConcept.campaign_id == campaign_id)
    )).scalars().all()

    for c in all_concepts:
        c.status = ConceptStatus.SELECTED if c.id == payload.concept_id else ConceptStatus.REJECTED

    campaign.selected_concept_id = payload.concept_id
    campaign.status = CampaignStatus.GENERATING_ASSETS
    await db.flush()

    # Run planning graph synchronously (it's CPU-only, fast, < 100ms)
    brand = (await db.execute(select(Brand).where(Brand.id == campaign.brand_id))).scalar_one()
    product = (await db.execute(select(Product).where(Product.id == campaign.product_id))).scalar_one()

    brand_context = {
        "name": brand.name,
        "description": brand.description,
        "primary_colors": brand.primary_colors,
        "secondary_colors": brand.secondary_colors,
        "fonts": brand.fonts,
        "tone": brand.tone,
    }
    product_context = {
        "name": product.name,
        "description": product.description,
        "image_urls": product.product_images,
    }

    # Serialize concept using ad_copy key for graph layer
    concept_dict: dict[str, Any] = {
        "name": concept.name,
        "description": concept.description,
        "visual_dna": concept.visual_dna,
        "ad_copy": concept.copy,   # translate DB column -> graph key
        "rationale": concept.rationale,
    }

    planner = build_planning_graph()
    planner_state = PlannerState(
        concept=concept_dict,
        brand_context=brand_context,
        product_context=product_context,
        asset_specs=[],
    )
    result = planner.invoke(planner_state)
    asset_specs = result["asset_specs"]

    # Create 3 fixed CreativeAsset rows, one per placement spec
    asset_ids: list[uuid.UUID] = []
    for spec in asset_specs:
        asset = CreativeAsset(
            campaign_id=campaign_id,
            concept_id=payload.concept_id,
            platform=Platform(spec["platform"]),
            placement=Placement(spec["placement"]),
            aspect_ratio=spec["aspect_ratio"],
            width=spec["width"],
            height=spec["height"],
            asset_spec=spec,
            status=AssetStatus.PENDING,
        )
        db.add(asset)
        await db.flush()
        asset_ids.append(asset.id)

    # Commit explicitly so asset rows are visible to background task sessions.
    await db.commit()

    # Schedule generation loop for each asset
    for asset_id in asset_ids:
        background.add_task(
            _run_generation_loop,
            asset_id=asset_id,
            product_images=product.product_images,
            brand_context=brand_context,
            session_factory=get_session_factory(),
        )

    return campaign


class EvaluationOut(BaseModel):
    overall_score: float
    product_fidelity: float
    brand_consistency: float
    composition_score: float
    prompt_alignment: float
    passed: bool
    failure_reason: str | None


class AttemptOut(BaseModel):
    attempt_number: int
    image_url: str | None
    infra_failed: bool
    evaluation: EvaluationOut | None


class AssetOut(BaseModel):
    id: uuid.UUID
    platform: str
    placement: str
    aspect_ratio: str
    status: AssetStatus
    attempts: list[AttemptOut]


@router.get("/campaigns/{campaign_id}/assets", response_model=list[AssetOut])
async def list_assets(
    campaign_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Poll target for the generation phase. Each asset's `attempts` list grows
    as the generate -> evaluate -> diagnose -> regenerate loop runs, up to
    MAX_ATTEMPTS, so the frontend can show the improvement-over-attempts
    view described in the spec (this is the strongest demo moment).
    """
    # Verify campaign ownership
    campaign = (await db.execute(
        select(Campaign).where(Campaign.id == campaign_id, Campaign.owner_id == user.id)
    )).scalar_one_or_none()
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")

    # Load assets with their attempts and evaluations in one query
    assets = (await db.execute(
        select(CreativeAsset)
        .where(CreativeAsset.campaign_id == campaign_id)
        .options(
            selectinload(CreativeAsset.attempts).selectinload(GenerationAttempt.evaluation)
        )
        .order_by(CreativeAsset.created_at)
    )).scalars().all()

    out = []
    for asset in assets:
        attempts_out = []
        for attempt in asset.attempts:
            eval_out = None
            if attempt.evaluation:
                e = attempt.evaluation
                eval_out = EvaluationOut(
                    overall_score=e.overall_score,
                    product_fidelity=e.product_fidelity,
                    brand_consistency=e.brand_consistency,
                    composition_score=e.composition_score,
                    prompt_alignment=e.prompt_alignment,
                    passed=e.passed,
                    failure_reason=e.failure_reason,
                )
            attempts_out.append(AttemptOut(
                attempt_number=attempt.attempt_number,
                image_url=attempt.image_url,
                infra_failed=attempt.infra_failed,
                evaluation=eval_out,
            ))
        out.append(AssetOut(
            id=asset.id,
            platform=asset.platform.value,
            placement=asset.placement.value,
            aspect_ratio=asset.aspect_ratio,
            status=asset.status,
            attempts=attempts_out,
        ))
    return out


@router.post("/campaigns/{campaign_id}/assets/{asset_id}/regenerate", response_model=AssetOut, status_code=202)
async def manual_regenerate(
    campaign_id: uuid.UUID,
    asset_id: uuid.UUID,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Manual override for assets stuck in MANUAL_REVIEW — resets the attempt
    counter and runs one fresh generation attempt. Distinct from the
    automatic retry loop, which stops itself at MAX_ATTEMPTS.
    """
    # Verify campaign ownership
    campaign = (await db.execute(
        select(Campaign).where(Campaign.id == campaign_id, Campaign.owner_id == user.id)
    )).scalar_one_or_none()
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")

    asset = (await db.execute(
        select(CreativeAsset)
        .where(CreativeAsset.id == asset_id, CreativeAsset.campaign_id == campaign_id)
        .options(selectinload(CreativeAsset.attempts).selectinload(GenerationAttempt.evaluation))
    )).scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")

    if asset.status != AssetStatus.MANUAL_REVIEW:
        raise HTTPException(
            status_code=409,
            detail=f"Asset is not in manual_review state (current: {asset.status.value}).",
        )

    brand = (await db.execute(select(Brand).where(Brand.id == campaign.brand_id))).scalar_one()
    product = (await db.execute(select(Product).where(Product.id == campaign.product_id))).scalar_one()

    brand_context = {
        "name": brand.name, "description": brand.description,
        "primary_colors": brand.primary_colors, "secondary_colors": brand.secondary_colors,
        "fonts": brand.fonts, "tone": brand.tone,
    }

    from app.database import get_session_factory as _gsf
    background.add_task(
        _run_generation_loop,
        asset_id=asset_id,
        product_images=product.product_images,
        brand_context=brand_context,
        is_manual_retry=True,
        session_factory=_gsf(),
    )

    # Return current state (async — client polls)
    attempts_out = [
        AttemptOut(
            attempt_number=a.attempt_number,
            image_url=a.image_url,
            infra_failed=a.infra_failed,
            evaluation=EvaluationOut(
                overall_score=a.evaluation.overall_score,
                product_fidelity=a.evaluation.product_fidelity,
                brand_consistency=a.evaluation.brand_consistency,
                composition_score=a.evaluation.composition_score,
                prompt_alignment=a.evaluation.prompt_alignment,
                passed=a.evaluation.passed,
                failure_reason=a.evaluation.failure_reason,
            ) if a.evaluation else None,
        )
        for a in asset.attempts
    ]
    return AssetOut(
        id=asset.id,
        platform=asset.platform.value,
        placement=asset.placement.value,
        aspect_ratio=asset.aspect_ratio,
        status=asset.status,
        attempts=attempts_out,
    )


# ---------------------------------------------------------------------------
# Background tasks (called via BackgroundTasks above, not exposed as routes)
# ---------------------------------------------------------------------------

async def _run_director(
    campaign_id: uuid.UUID,
    brand_context: dict,
    product_context: dict,
    brief_text: str,
    target_audience: str | None,
    session_factory=None,
) -> None:
    """
    Runs the Director LangGraph, persists resulting CreativeConcept rows,
    and flips campaign.status -> concepts_ready.
    Opens its own DB session (background tasks run outside the request session).
    session_factory: injectable for tests; defaults to AsyncSessionLocal.
    """
    if session_factory is None:
        from app.database import AsyncSessionLocal
        session_factory = AsyncSessionLocal

    async with session_factory() as db:
        try:
            director = build_director_graph()
            init_state = DirectorState(
                brand_context=brand_context,
                product_context=product_context,
                brief_text=brief_text,
                target_audience=target_audience,
                concepts=[],
            )
            result = await director.ainvoke(init_state)
            concepts: list[dict] = result["concepts"]

            # Persist concepts — translate graph key `ad_copy` -> DB column `copy`
            for concept_data in concepts:
                concept = CreativeConcept(
                    campaign_id=campaign_id,
                    name=concept_data["name"],
                    description=concept_data["description"],
                    visual_dna=concept_data["visual_dna"],
                    copy=concept_data["ad_copy"],   # graph key -> DB column
                    rationale=concept_data["rationale"],
                    status=ConceptStatus.PROPOSED,
                )
                db.add(concept)

            campaign = (await db.execute(
                select(Campaign).where(Campaign.id == campaign_id)
            )).scalar_one()
            campaign.status = CampaignStatus.CONCEPTS_READY
            await db.commit()
            logger.info("_run_director: campaign=%s -> concepts_ready (%d concepts)", campaign_id, len(concepts))

        except Exception as exc:
            logger.exception("_run_director: failed for campaign=%s: %s", campaign_id, exc)
            try:
                campaign = (await db.execute(
                    select(Campaign).where(Campaign.id == campaign_id)
                )).scalar_one()
                campaign.status = CampaignStatus.FAILED
                await db.commit()
            except Exception:
                pass


async def _run_generation_loop(
    asset_id: uuid.UUID,
    product_images: list[str],
    brand_context: dict,
    is_manual_retry: bool = False,
    session_factory=None,
) -> None:
    """
    Checkpoint 2 implementation: single attempt per asset (happy path).

    Design decision: We implement ONE attempt here (not the full 3-attempt
    retry loop) because:
    1. MockImageProvider always passes — retries add zero test coverage value.
    2. The retry loop's business value (showing degraded -> improved outputs)
       only manifests with real providers (Checkpoint 3+).
    3. The full loop skeleton is already specified in the route docstring
       and in graph.py comments — it is not lost, just deferred.
    4. Keeping this simple avoids the complexity of session-per-attempt
       timing when both generation and evaluation are <1ms mocks.

    The loop will be fully implemented in Checkpoint 3 when RemoteFluxKaggle-
    Provider and the real VisionEvaluator are wired in.
    session_factory: injectable for tests; defaults to AsyncSessionLocal.
    """
    if session_factory is None:
        from app.database import AsyncSessionLocal
        session_factory = AsyncSessionLocal

    image_provider = get_image_provider()
    vision_evaluator = get_vision_evaluator()
    provider_name = os.environ.get("IMAGE_GENERATION_PROVIDER", "mock")

    async with session_factory() as db:
        try:
            asset = (await db.execute(
                select(CreativeAsset).where(CreativeAsset.id == asset_id)
            )).scalar_one()

            asset.status = AssetStatus.GENERATING
            await db.flush()

            # Determine attempt number (manual retry continues from prior attempts)
            existing_count = len((await db.execute(
                select(GenerationAttempt).where(GenerationAttempt.asset_id == asset_id)
            )).scalars().all())
            attempt_number = existing_count + 1

            # Run single generation attempt through the graph nodes
            gen_state = GenerationState(
                asset_spec=asset.asset_spec,
                product_images=product_images,
                reference_images=[],         # product images serve as reference in V1
                brand_context=brand_context,
                corrective_instruction=None,
                generated_image_url=None,
                scores=None,
                outcome=None,
            )

            gen_state = await generate_image_node(gen_state, image_provider)

            if gen_state["outcome"] == "infra_failed":
                attempt = GenerationAttempt(
                    asset_id=asset_id,
                    attempt_number=attempt_number,
                    prompt_used=asset.asset_spec.get("generation_prompt", ""),
                    corrective_instruction=None,
                    image_url=None,
                    provider=provider_name,
                    infra_failed=True,
                    infra_error="Provider raised an exception",
                )
                db.add(attempt)
                asset.status = AssetStatus.INFRA_FAILED
                await db.commit()
                return

            gen_state = await evaluate_node(gen_state, vision_evaluator)

            prompt_used = asset.asset_spec.get("generation_prompt", "")
            attempt = GenerationAttempt(
                asset_id=asset_id,
                attempt_number=attempt_number,
                prompt_used=prompt_used,
                corrective_instruction=None,
                image_url=gen_state["generated_image_url"],
                provider=provider_name,
                infra_failed=False,
            )
            db.add(attempt)
            await db.flush()

            scores = gen_state["scores"]
            evaluation = Evaluation(
                attempt_id=attempt.id,
                vlm_product_score=scores["vlm_product_score"],
                siglip_similarity=scores["siglip_similarity"],
                ocr_text_score=scores["ocr_text_score"],
                product_fidelity=scores["product_fidelity"],
                brand_consistency=scores["brand_consistency"],
                composition_score=scores["composition_score"],
                prompt_alignment=scores["prompt_alignment"],
                overall_score=scores["overall_score"],
                critical_text_error=scores["critical_text_error"],
                passed=scores["passed"],
                failure_reason=scores["failure_reason"],
                vlm_provider=provider_name,
                raw_response=scores,
            )
            db.add(evaluation)
            await db.flush()

            if gen_state["outcome"] == "approved":
                asset.status = AssetStatus.APPROVED
                asset.approved_attempt_id = attempt.id
            else:
                # With mock, this branch is unreachable; with real providers this
                # would loop. Full retry loop implemented in Checkpoint 3.
                asset.status = AssetStatus.MANUAL_REVIEW

            await db.commit()
            logger.info(
                "_run_generation_loop: asset=%s -> %s (attempt %d)",
                asset_id, asset.status.value, attempt_number,
            )

            # After all assets finish, check if campaign is complete
            await _maybe_complete_campaign(db, asset.campaign_id)

        except Exception as exc:
            logger.exception("_run_generation_loop: failed for asset=%s: %s", asset_id, exc)
            try:
                asset = (await db.execute(
                    select(CreativeAsset).where(CreativeAsset.id == asset_id)
                )).scalar_one()
                asset.status = AssetStatus.INFRA_FAILED
                await db.commit()
            except Exception:
                pass


async def _maybe_complete_campaign(db: AsyncSession, campaign_id: uuid.UUID) -> None:
    """
    If all 3 assets for this campaign are in a terminal state
    (APPROVED, MANUAL_REVIEW, or INFRA_FAILED), flip campaign -> COMPLETE.
    """
    terminal = {AssetStatus.APPROVED, AssetStatus.MANUAL_REVIEW, AssetStatus.INFRA_FAILED}
    assets = (await db.execute(
        select(CreativeAsset).where(CreativeAsset.campaign_id == campaign_id)
    )).scalars().all()

    if all(a.status in terminal for a in assets):
        campaign = (await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )).scalar_one()
        campaign.status = CampaignStatus.COMPLETE
        await db.commit()
        logger.info("_maybe_complete_campaign: campaign=%s -> complete", campaign_id)
