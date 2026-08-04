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
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.models import CampaignStatus, AssetStatus

router = APIRouter()

MAX_ATTEMPTS = 3  # 1 initial + 2 regenerations, per frozen retry policy


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


@router.post("/brands", response_model=BrandOut, status_code=201)
async def create_brand(payload: BrandCreate, user=Depends(...)):
    ...


class ProductCreate(BaseModel):
    brand_id: uuid.UUID
    name: str
    description: str | None = None
    # product_images uploaded via multipart on POST /products/{id}/images


class ProductOut(ProductCreate):
    id: uuid.UUID
    product_images: list[str]


@router.post("/products", response_model=ProductOut, status_code=201)
async def create_product(payload: ProductCreate, user=Depends(...)):
    ...


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


@router.post("/campaigns", response_model=CampaignOut, status_code=202)
async def create_campaign(payload: CampaignCreate, background: BackgroundTasks, user=Depends(...)):
    """
    Creates the Campaign row (status=generating_concepts) and schedules the
    Director graph as a background task. Client polls GET /campaigns/{id}
    until status flips to concepts_ready.
    """
    ...


class ConceptOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    visual_dna: dict
    copy: dict
    rationale: str
    status: str


@router.get("/campaigns/{campaign_id}", response_model=CampaignOut)
async def get_campaign(campaign_id: uuid.UUID, user=Depends(...)):
    ...


@router.get("/campaigns/{campaign_id}/concepts", response_model=list[ConceptOut])
async def list_concepts(campaign_id: uuid.UUID, user=Depends(...)):
    """Populated once campaign.status == concepts_ready."""
    ...


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
    user=Depends(...),
):
    """
    Marks the chosen CreativeConcept selected, the rest rejected. Creates
    the 3 fixed CreativeAsset rows (pending), runs the Planner graph to
    populate their asset_spec, then schedules generation for each asset.
    Sets campaign.status = generating_assets.
    """
    ...


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
async def list_assets(campaign_id: uuid.UUID, user=Depends(...)):
    """
    Poll target for the generation phase. Each asset's `attempts` list grows
    as the generate -> evaluate -> diagnose -> regenerate loop runs, up to
    MAX_ATTEMPTS, so the frontend can show the improvement-over-attempts
    view described in the spec (this is the strongest demo moment).
    """
    ...


@router.post("/campaigns/{campaign_id}/assets/{asset_id}/regenerate", response_model=AssetOut, status_code=202)
async def manual_regenerate(
    campaign_id: uuid.UUID,
    asset_id: uuid.UUID,
    background: BackgroundTasks,
    user=Depends(...),
):
    """
    Manual override for assets stuck in MANUAL_REVIEW — resets the attempt
    counter and runs one fresh generation attempt. Distinct from the
    automatic retry loop, which stops itself at MAX_ATTEMPTS.
    """
    ...


# ---------------------------------------------------------------------------
# Background task shape (called via BackgroundTasks above, not exposed as
# a route). Sketched here so the retry-cap logic lives next to its contract.
# ---------------------------------------------------------------------------

async def run_generation_loop(asset_id: uuid.UUID):
    """
    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        run generation_graph (generate -> evaluate[-> diagnose])
        persist GenerationAttempt + Evaluation rows
        if outcome == "approved":
            asset.status = APPROVED; asset.approved_attempt_id = attempt.id; return
        if outcome == "infra_failed":
            asset.status = INFRA_FAILED; return   # separate Kafka retry/DLQ policy post-V1, not a quality retry
        # else "retry": loop continues with corrective_instruction set
    asset.status = MANUAL_REVIEW
    """
    ...
