"""
LangGraph shape for the V1 vertical slice.

Two separate, short-lived graphs rather than one long-running interrupted
graph — matches the "deliberately boring" V1 philosophy. Orchestration
(when each graph runs) is FastAPI's job, not LangGraph's:

    POST /campaigns              -> run director_graph once
    POST /campaigns/{id}/select  -> run planning_graph once
    (generation + eval loop)     -> run generation_graph per asset,
                                     looped by FastAPI up to 3 attempts

Responsibilities stay separated per the frozen constitution:
    LangGraph = reasoning (concepts, specs, diagnosis)
    Provider abstractions = execution (image generation, vision evaluation)
    FastAPI background tasks = scheduling (Kafka replaces this post-V1)
"""
from __future__ import annotations

import abc
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# Provider abstractions — swappable, mocked by default in dev
# ---------------------------------------------------------------------------

class ImageGenerationProvider(abc.ABC):
    @abc.abstractmethod
    async def generate(
        self,
        product_images: list[str],
        reference_images: list[str],
        prompt: str,
        width: int,
        height: int,
    ) -> str:
        """Returns a URL/path to the generated image. Raises on infra failure."""
        ...


class MockImageProvider(ImageGenerationProvider):
    async def generate(self, product_images, reference_images, prompt, width, height) -> str:
        return "/local/mock/generated_placeholder.jpg"


class RemoteFluxKaggleProvider(ImageGenerationProvider):
    """
    Primary DEV/BUILD provider — $0 cost. Posts a job to the local
    generation-gateway which Kaggle polls (or, for the simplest V1 dev loop,
    calls a tunneled Kaggle HTTPS endpoint directly). Implementation detail
    is intentionally not fixed here — the contract (inputs -> image url) is
    what the rest of the system depends on.

    Use this for all day-to-day development, testing the pipeline, and the
    generate/evaluate/retry loop. Quality is "good enough to prove the
    architecture works," not "portfolio-video good."
    """
    async def generate(self, product_images, reference_images, prompt, width, height) -> str:
        raise NotImplementedError("Wire up once the Kaggle worker gateway exists")


class GeminiImageProvider(ImageGenerationProvider):
    """
    SHOWCASE-ONLY provider — paid API, used exclusively when recording the
    portfolio demo video. Same ImageGenerationProvider contract as the free
    Kaggle/FLUX path, so swapping providers is a one-line env var change and
    touches no other code:

        IMAGE_GENERATION_PROVIDER=gemini

    Do not default to this provider anywhere, do not call it from tests or
    CI, and do not leave it enabled after recording. The architecture story
    for interviews is: "generation is provider-agnostic; FLUX.2 Klein on
    free Kaggle GPU is the deployed reference implementation, and a paid
    provider can be swapped in with no application-layer changes when
    higher quality is worth the cost" — this class is the proof of that
    claim, not the production default.
    """
    async def generate(self, product_images, reference_images, prompt, width, height) -> str:
        raise NotImplementedError("Wire up only when preparing the showcase demo recording")


class VisionEvaluator(abc.ABC):
    @abc.abstractmethod
    async def evaluate(
        self,
        original_product_images: list[str],
        generated_image: str,
        asset_spec: dict,
        brand_context: dict,
    ) -> dict:
        """
        Returns dict matching the Evaluation model's scoring fields:
        vlm_product_score, siglip_similarity, ocr_text_score,
        product_fidelity, brand_consistency, composition_score,
        prompt_alignment, overall_score, critical_text_error,
        passed, failure_reason
        """
        ...


class MockVisionEvaluator(VisionEvaluator):
    async def evaluate(self, original_product_images, generated_image, asset_spec, brand_context) -> dict:
        return {
            "vlm_product_score": 0.9, "siglip_similarity": 0.88, "ocr_text_score": 1.0,
            "product_fidelity": 0.9, "brand_consistency": 0.9, "composition_score": 0.9,
            "prompt_alignment": 0.9, "overall_score": 0.9,
            "critical_text_error": False, "passed": True, "failure_reason": None,
        }


# Pass thresholds — hard constraints on top of the weighted average, per spec.
# A beautiful image with a mangled product must still fail.
PASS_THRESHOLDS = {
    "overall_score": 0.90,
    "product_fidelity": 0.92,
    "brand_consistency": 0.85,
}


def evaluate_pass(scores: dict) -> tuple[bool, str | None]:
    if scores["critical_text_error"]:
        return False, "critical_text_error"
    for field, threshold in PASS_THRESHOLDS.items():
        if scores[field] < threshold:
            return False, f"{field}_below_threshold ({scores[field]:.2f} < {threshold})"
    return True, None


# ---------------------------------------------------------------------------
# Graph 1: Director — campaign brief -> 2-3 CreativeConcepts
# ---------------------------------------------------------------------------

class DirectorState(TypedDict):
    brand_context: dict          # {name, description, colors, fonts, tone}
    product_context: dict        # {name, description, image_urls}
    brief_text: str
    target_audience: str | None
    concepts: list[dict]         # populated by generate_concepts node


def generate_concepts_node(state: DirectorState) -> DirectorState:
    """
    Single LLM call producing 2-3 CreativeConcept payloads (name, description,
    visual_dna, copy, rationale) as structured JSON. Implementation calls the
    chosen LLM with a system prompt that fixes the exact schema documented in
    models.CreativeConcept and requests strict JSON output.
    """
    # Placeholder — real implementation invokes the LLM here.
    state["concepts"] = []
    return state


def build_director_graph():
    graph = StateGraph(DirectorState)
    graph.add_node("generate_concepts", generate_concepts_node)
    graph.set_entry_point("generate_concepts")
    graph.add_edge("generate_concepts", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Graph 2: Asset Planner — selected concept -> 3 fixed AssetSpecs
# ---------------------------------------------------------------------------

FIXED_PLACEMENTS = [
    {"platform": "instagram", "placement": "ig_feed", "aspect_ratio": "4:5", "width": 1080, "height": 1350},
    {"platform": "instagram", "placement": "ig_story", "aspect_ratio": "9:16", "width": 1080, "height": 1920},
    {"platform": "website", "placement": "website_hero", "aspect_ratio": "16:9", "width": 1920, "height": 1080},
]


class PlannerState(TypedDict):
    concept: dict          # the selected CreativeConcept, serialized
    brand_context: dict
    product_context: dict
    asset_specs: list[dict]  # populated by plan_assets node


def plan_assets_node(state: PlannerState) -> PlannerState:
    """
    For each of the 3 fixed placements, derive composition rules
    (product_position, product_scale, copy_safe_area, cta_safe_area,
    framing) from the concept's visual_dna, then render a model-specific
    generation_prompt. AssetSpec stays canonical; generation_prompt is
    derived so swapping FLUX later doesn't touch this schema.
    """
    specs = []
    for p in FIXED_PLACEMENTS:
        specs.append({
            **p,
            "subject": {"product_position": None, "product_scale": None, "preserve_identity": True},
            "environment": state["concept"]["visual_dna"].get("environment"),
            "lighting": state["concept"]["visual_dna"].get("lighting"),
            "composition": {"framing": None, "negative_space": None, "copy_safe_area": None, "cta_safe_area": None},
            "style": state["concept"]["visual_dna"].get("mood", []),
            "generation_prompt": None,  # built from the fields above
        })
    state["asset_specs"] = specs
    return state


def build_planning_graph():
    graph = StateGraph(PlannerState)
    graph.add_node("plan_assets", plan_assets_node)
    graph.set_entry_point("plan_assets")
    graph.add_edge("plan_assets", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Graph 3: Generation + evaluation loop — one AssetSpec -> approved asset
#           or MANUAL_REVIEW. Looped externally by FastAPI up to 3 attempts;
#           this graph handles a single attempt + diagnosis for the next one.
# ---------------------------------------------------------------------------

class GenerationState(TypedDict):
    asset_spec: dict
    product_images: list[str]
    reference_images: list[str]
    brand_context: dict
    corrective_instruction: str | None   # set by diagnose_failure on retry
    generated_image_url: str | None
    scores: dict | None
    outcome: Literal["approved", "retry", "manual_review", "infra_failed"] | None


async def generate_image_node(state: GenerationState, provider: ImageGenerationProvider) -> GenerationState:
    prompt = state["asset_spec"]["generation_prompt"]
    if state["corrective_instruction"]:
        prompt = f"{prompt}\n\nCorrection: {state['corrective_instruction']}"
    try:
        state["generated_image_url"] = await provider.generate(
            product_images=state["product_images"],
            reference_images=state["reference_images"],
            prompt=prompt,
            width=state["asset_spec"]["width"],
            height=state["asset_spec"]["height"],
        )
    except Exception:
        state["outcome"] = "infra_failed"
    return state


async def evaluate_node(state: GenerationState, evaluator: VisionEvaluator) -> GenerationState:
    if state.get("outcome") == "infra_failed":
        return state
    scores = await evaluator.evaluate(
        original_product_images=state["product_images"],
        generated_image=state["generated_image_url"],
        asset_spec=state["asset_spec"],
        brand_context=state["brand_context"],
    )
    passed, reason = evaluate_pass(scores)
    scores["passed"] = passed
    scores["failure_reason"] = reason
    state["scores"] = scores
    state["outcome"] = "approved" if passed else "retry"
    return state


def diagnose_failure_node(state: GenerationState) -> GenerationState:
    """Turns failure_reason into a corrective instruction for the next attempt."""
    reason = state["scores"]["failure_reason"] if state.get("scores") else "unknown"
    state["corrective_instruction"] = f"Fix: {reason}"  # real impl: LLM-authored correction
    return state


# Note: build_generation_graph wires generate -> evaluate -> (END | diagnose),
# with attempt-count/3-attempt cap enforced by the FastAPI caller, since that
# cap is a business rule tied to CreativeAsset/GenerationAttempt rows, not
# graph-internal state.
