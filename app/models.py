"""
Domain models for the Agentic Creative Campaign Engine — V1 vertical slice.

Scope frozen per project spec:
- Brand/Product = plain Postgres + local disk. No Qdrant, no embeddings yet.
- No Kafka yet — generation attempts are tracked synchronously via background tasks.
- Every generation attempt is stored (not overwritten) to show the
  generate -> evaluate -> diagnose -> correct loop in the demo.
- Retry cap: 2 regenerations / 3 total attempts per asset, then MANUAL_REVIEW.
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    ForeignKey,
    String,
    Text,
    Enum as SAEnum,
    DateTime,
    Float,
    Boolean,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CampaignStatus(str, enum.Enum):
    DRAFT = "draft"                        # created, brief saved
    GENERATING_CONCEPTS = "generating_concepts"   # Director running
    CONCEPTS_READY = "concepts_ready"      # awaiting user selection
    GENERATING_ASSETS = "generating_assets"  # Planner + FLUX running
    COMPLETE = "complete"                  # all assets approved or manual_review
    FAILED = "failed"                      # unrecoverable infra failure


class ConceptStatus(str, enum.Enum):
    PROPOSED = "proposed"
    SELECTED = "selected"
    REJECTED = "rejected"


class AssetStatus(str, enum.Enum):
    PENDING = "pending"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    MANUAL_REVIEW = "manual_review"
    INFRA_FAILED = "infra_failed"   # worker/generation crash, not a quality failure


class Platform(str, enum.Enum):
    INSTAGRAM = "instagram"
    WEBSITE = "website"
    # V1 is frozen to these two platforms (3 assets total). Extend later.


class Placement(str, enum.Enum):
    IG_FEED = "ig_feed"       # 4:5
    IG_STORY = "ig_story"     # 9:16
    WEBSITE_HERO = "website_hero"  # 16:9


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class User(Base):
    """
    Mirrors identity established by NextAuth.js (Google provider) on the
    frontend. FastAPI verifies the NextAuth-issued JWT and upserts this row
    keyed on google_sub. This table is the source of truth for ownership,
    not for session management.
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    google_sub: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    brands: Mapped[list["Brand"]] = relationship(back_populates="owner")


# ---------------------------------------------------------------------------
# Brand / Product — plain application data, no vector store
# ---------------------------------------------------------------------------

class Brand(Base):
    __tablename__ = "brands"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)

    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    logo_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)  # local disk path/URL in V1

    primary_colors: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    secondary_colors: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    fonts: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    tone: Mapped[str | None] = mapped_column(Text, nullable=True)  # positioning/voice description

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    owner: Mapped["User"] = relationship(back_populates="brands")
    products: Mapped[list["Product"]] = relationship(back_populates="brand", cascade="all, delete-orphan")
    campaigns: Mapped[list["Campaign"]] = relationship(back_populates="brand")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = _uuid_pk()
    brand_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("brands.id"), index=True)

    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # V1: same product, multiple angles. Simple string list of local paths/URLs
    # is sufficient — no separate ProductImage table until multi-image handling
    # gets complex (per the frozen "single image if complex" fallback).
    product_images: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    extra_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)  # freeform: category, SKU, etc.

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    brand: Mapped["Brand"] = relationship(back_populates="products")


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[uuid.UUID] = _uuid_pk()
    brand_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("brands.id"), index=True)
    product_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("products.id"), index=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)

    # Raw brief as entered by the user. Kept even though the Director turns it
    # into structured concepts — useful for re-running/regenerating concepts.
    brief_text: Mapped[str] = mapped_column(Text)
    target_audience: Mapped[str | None] = mapped_column(String(500), nullable=True)

    status: Mapped[CampaignStatus] = mapped_column(
        SAEnum(CampaignStatus, name="campaign_status"), default=CampaignStatus.DRAFT
    )
    selected_concept_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("creative_concepts.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    brand: Mapped["Brand"] = relationship(back_populates="campaigns")
    concepts: Mapped[list["CreativeConcept"]] = relationship(
        back_populates="campaign",
        foreign_keys="CreativeConcept.campaign_id",
        cascade="all, delete-orphan",
    )
    assets: Mapped[list["CreativeAsset"]] = relationship(back_populates="campaign", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# CreativeConcept — output of the Director node
# ---------------------------------------------------------------------------

class CreativeConcept(Base):
    __tablename__ = "creative_concepts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    campaign_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("campaigns.id"), index=True)

    name: Mapped[str] = mapped_column(String(255))          # "Midnight Encounter"
    description: Mapped[str] = mapped_column(Text)          # the actual creative idea, human-readable

    # Structured Visual DNA — kept as JSON rather than normalized tables.
    # Shape: { palette: [str], lighting: str, environment: str,
    #          materials: [str], mood: [str], photography_style: str }
    visual_dna: Mapped[dict] = mapped_column(JSONB)

    # Shape: { headline: str, subcopy: str | None, cta: str | None }
    copy: Mapped[dict] = mapped_column(JSONB)

    rationale: Mapped[str] = mapped_column(Text)  # why this concept fits the brief
    status: Mapped[ConceptStatus] = mapped_column(
        SAEnum(ConceptStatus, name="concept_status"), default=ConceptStatus.PROPOSED
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    campaign: Mapped["Campaign"] = relationship(back_populates="concepts", foreign_keys=[campaign_id])


# ---------------------------------------------------------------------------
# CreativeAsset — one per (campaign, placement). Fixed 3 per campaign in V1.
# ---------------------------------------------------------------------------

class CreativeAsset(Base):
    __tablename__ = "creative_assets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    campaign_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("campaigns.id"), index=True)
    concept_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("creative_concepts.id"), index=True)

    platform: Mapped[Platform] = mapped_column(SAEnum(Platform, name="platform"))
    placement: Mapped[Placement] = mapped_column(SAEnum(Placement, name="placement"))
    aspect_ratio: Mapped[str] = mapped_column(String(10))   # "4:5" | "9:16" | "16:9"
    width: Mapped[int] = mapped_column()
    height: Mapped[int] = mapped_column()

    # Output of the Asset Planner. Canonical spec — generation_prompt inside
    # is DERIVED, not the source of truth, so swapping FLUX for another model
    # only changes the prompt-builder, not this record.
    # Shape: { subject: {...}, environment: str, lighting: str,
    #          composition: {...}, style: [str], generation_prompt: str }
    asset_spec: Mapped[dict] = mapped_column(JSONB)

    status: Mapped[AssetStatus] = mapped_column(
        SAEnum(AssetStatus, name="asset_status"), default=AssetStatus.PENDING
    )
    approved_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("generation_attempts.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    campaign: Mapped["Campaign"] = relationship(back_populates="assets")
    attempts: Mapped[list["GenerationAttempt"]] = relationship(
        back_populates="asset",
        foreign_keys="GenerationAttempt.asset_id",
        cascade="all, delete-orphan",
        order_by="GenerationAttempt.attempt_number",
    )


# ---------------------------------------------------------------------------
# GenerationAttempt — every attempt stored, never overwritten (max 3)
# ---------------------------------------------------------------------------

class GenerationAttempt(Base):
    __tablename__ = "generation_attempts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("creative_assets.id"), index=True)

    attempt_number: Mapped[int] = mapped_column()  # 1, 2, or 3

    # Prompt actually sent to the provider this attempt (may include a
    # corrective addendum from a prior failed evaluation).
    prompt_used: Mapped[str] = mapped_column(Text)
    corrective_instruction: Mapped[str | None] = mapped_column(Text, nullable=True)

    image_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)  # local disk path in V1
    provider: Mapped[str] = mapped_column(String(50))  # "mock" | "flux2_klein_kaggle"

    infra_failed: Mapped[bool] = mapped_column(Boolean, default=False)
    infra_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    asset: Mapped["CreativeAsset"] = relationship(back_populates="attempts", foreign_keys=[asset_id])
    evaluation: Mapped["Evaluation | None"] = relationship(back_populates="attempt", uselist=False)


# ---------------------------------------------------------------------------
# Evaluation — one per GenerationAttempt
# ---------------------------------------------------------------------------

class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("generation_attempts.id"), unique=True, index=True)

    # Component scores, 0.0-1.0
    vlm_product_score: Mapped[float] = mapped_column(Float)
    siglip_similarity: Mapped[float] = mapped_column(Float)
    ocr_text_score: Mapped[float] = mapped_column(Float)

    product_fidelity: Mapped[float] = mapped_column(Float)     # weighted: vlm 40 / siglip 35 / ocr 25
    brand_consistency: Mapped[float] = mapped_column(Float)
    composition_score: Mapped[float] = mapped_column(Float)
    prompt_alignment: Mapped[float] = mapped_column(Float)

    overall_score: Mapped[float] = mapped_column(Float)
    critical_text_error: Mapped[bool] = mapped_column(Boolean, default=False)

    passed: Mapped[bool] = mapped_column(Boolean)  # result of hard-constraint pass logic, not just averaging
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)  # diagnosis, feeds next attempt's corrective_instruction

    vlm_provider: Mapped[str] = mapped_column(String(50))  # "mock" | "gemini" | ...
    raw_response: Mapped[dict] = mapped_column(JSONB, default=dict)  # full VLM response for debugging/demo

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    attempt: Mapped["GenerationAttempt"] = relationship(back_populates="evaluation")
