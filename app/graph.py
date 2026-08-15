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

LLM Provider (concept generation)
----------------------------------
Controlled by env var LLM_PROVIDER (default: "groq").
Supported:
  "groq" — Groq (llama-3.3-70b-versatile by default) via the groq SDK.
            Fast + free tier with good structured JSON output. This is the
            current default for the Director's reasoning.
  "gemini" — Google Gemini Flash via google-generativeai SDK (requires
              GEMINI_API_KEY env var). Kept as a swappable option.
Swappable to any other provider by implementing a parallel LLMProvider
abstract class (not added yet — one provider is sufficient for V1).
"""
from __future__ import annotations

import abc
import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import TypedDict, Literal

import httpx
from langgraph.graph import StateGraph, END

from app.config import settings

logger = logging.getLogger(__name__)


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
    Primary DEV/BUILD provider — $0 cost via Kaggle T4 running FLUX.2 Klein.

    V1 contract: direct HTTPS call to a tunneled Kaggle endpoint (the
    "simplest V1 dev loop" from the constitution; the gateway/pull pattern
    is build-later). The worker is expected to accept:

        POST {KAGGLE_GATEWAY_URL}/generate
        Authorization: Bearer {KAGGLE_API_KEY}   (only if configured)
        {
          "product_images":   [urls/paths...],
          "reference_images": [urls/paths...],
          "prompt": "...",
          "width": 1080,
          "height": 1350
        }

    and respond either:
      - 200 JSON {"image": "<base64 png>"} (preferred — Kaggle can't host
        public URLs), or
      - 200 raw image/* bytes (simple workers).

    The image is decoded and written to local disk (MEDIA_DIR), because V1
    stores images on local disk, not S3. Returns the local path.

    Raises on any transport/HTTP/protocol failure so the caller treats it as
    an infra failure (INFRA_FAILED), not a quality failure.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.KAGGLE_GATEWAY_URL).rstrip("/")
        self.api_key = settings.KAGGLE_API_KEY if api_key is None else api_key
        self.timeout = settings.KAGGLE_REQUEST_TIMEOUT_SECONDS if timeout is None else timeout

    async def generate(
        self,
        product_images: list[str],
        reference_images: list[str],
        prompt: str,
        width: int,
        height: int,
    ) -> str:
        if not self.base_url:
            raise RuntimeError(
                "RemoteFluxKaggleProvider needs KAGGLE_GATEWAY_URL set to a tunneled "
                "Kaggle HTTPS endpoint (or use IMAGE_GENERATION_PROVIDER=mock)."
            )

        url = f"{self.base_url}/generate"
        payload = {
            "product_images": product_images,
            "reference_images": reference_images,
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return await self._persist(resp, width, height)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Kaggle generation HTTP error: {exc}") from exc

    async def _persist(self, resp: httpx.Response, width: int, height: int) -> str:
        """Decode the worker response into a local PNG file; return its path."""
        content_type = resp.headers.get("content-type", "")
        if "image/" in content_type:
            data = resp.content
        else:
            body = resp.json()
            if not isinstance(body, dict):
                data = resp.content
            elif body.get("image"):
                data = base64.b64decode(body["image"])
            elif body.get("image_url"):
                # Remote-hosted URL (rare on Kaggle); return it as-is.
                return body["image_url"]
            else:
                raise RuntimeError(f"Unexpected Kaggle response payload: {str(body)[:200]}")

        media_dir = Path(settings.MEDIA_DIR)
        media_dir.mkdir(parents=True, exist_ok=True)
        path = media_dir / f"{width}x{height}_{uuid.uuid4().hex}.png"
        path.write_bytes(data)
        logger.info("RemoteFluxKaggleProvider: saved %s (%d bytes)", path, len(data))
        return str(path)


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
            "vlm_product_score": 0.95, "siglip_similarity": 0.93, "ocr_text_score": 1.0,
            # product_fidelity must be >= PASS_THRESHOLDS["product_fidelity"] (0.85);
            # was 0.9 vs the old 0.92 threshold which failed eval and landed assets
            # in MANUAL_REVIEW.
            "product_fidelity": 0.93, "brand_consistency": 0.92, "composition_score": 0.91,
            "prompt_alignment": 0.92, "overall_score": 0.92,
            "critical_text_error": False, "passed": True, "failure_reason": None,
        }


# Pass thresholds — hard constraints on top of the weighted average, per spec.
# A beautiful image with a mangled product must still fail.
# product_fidelity: real FLUX scene-ads (styled composition, product re-lit/
# re-scaled vs the clean product shot) score ~0.81-0.92 on SigLIP/VLM; 0.92
# sent every asset to MANUAL_REVIEW. 0.85 still rejects visibly mangled
# products while letting good scene-ads pass (owner decision, 2026-08-15).
PASS_THRESHOLDS = {
    "overall_score": 0.90,
    "product_fidelity": 0.85,
    "brand_consistency": 0.85,
}


def evaluate_pass(scores: dict) -> tuple[bool, str | None]:
    if scores["critical_text_error"]:
        return False, "critical_text_error"
    for field, threshold in PASS_THRESHOLDS.items():
        if scores[field] < threshold:
            # 3 decimals so a real 0.9199 doesn't masquerade as "0.92 < 0.92".
            return False, f"{field}_below_threshold ({scores[field]:.3f} < {threshold})"
    return True, None


# ---------------------------------------------------------------------------
# LLM helpers — Groq (default) via the groq SDK; Gemini Flash via the
# google-generativeai SDK. Both return the raw concepts text, which is then
# parsed/validated by _parse_llm_json / _validate_concepts.
# ---------------------------------------------------------------------------

# JSON schema the Director LLM must produce for each concept.
CONCEPT_SCHEMA = {
    "type": "array",
    "minItems": 2,
    "maxItems": 3,
    "items": {
        "type": "object",
        "required": ["name", "description", "visual_dna", "ad_copy", "rationale"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "visual_dna": {
                "type": "object",
                "required": ["palette", "lighting", "environment", "materials", "mood", "photography_style"],
                "properties": {
                    "palette": {"type": "array", "items": {"type": "string"}},
                    "lighting": {"type": "string"},
                    "environment": {"type": "string"},
                    "materials": {"type": "array", "items": {"type": "string"}},
                    "mood": {"type": "array", "items": {"type": "string"}},
                    "photography_style": {"type": "string"},
                },
            },
            "ad_copy": {
                "type": "object",
                "required": ["headline", "subcopy", "cta"],
                "properties": {
                    "headline": {"type": "string"},
                    "subcopy": {"type": ["string", "null"]},
                    "cta": {"type": ["string", "null"]},
                },
            },
            "rationale": {"type": "string"},
        },
    },
}

DIRECTOR_SYSTEM_PROMPT = """\
You are a Creative Director at a world-class advertising agency.

Given a brand, product, and campaign brief, produce 2-3 creative campaign
concepts. Each concept must be distinct in visual style and emotional tone.

Respond with ONLY a JSON array (no markdown fences, no prose) matching this
exact structure for each element:

{
  "name": "Concept name",
  "description": "Full creative idea in 2-4 sentences",
  "visual_dna": {
    "palette": ["#hex1", "#hex2"],
    "lighting": "e.g. golden hour",
    "environment": "e.g. urban rooftop",
    "materials": ["glass", "steel"],
    "mood": ["aspirational", "bold"],
    "photography_style": "e.g. editorial"
  },
  "ad_copy": {
    "headline": "Short punchy headline",
    "subcopy": "Supporting copy or null",
    "cta": "Call to action or null"
  },
  "rationale": "Why this concept fits the brief"
}

Rules:
- Output a JSON array of 2-3 objects. Nothing else.
- All fields are required. Use null for optional fields you intentionally leave blank.
- Each concept must have a meaningfully different visual_dna (different palette, environment, mood).
"""


def _build_director_user_prompt(state: "DirectorState") -> str:
    b = state["brand_context"]
    p = state["product_context"]
    lines = [
        f"BRAND: {b.get('name', 'Unknown')}",
        f"Brand description: {b.get('description', 'Not provided')}",
        f"Primary colors: {', '.join(b.get('primary_colors', []))}",
        f"Secondary colors: {', '.join(b.get('secondary_colors', []))}",
        f"Fonts: {', '.join(b.get('fonts', []))}",
        f"Brand tone/voice: {b.get('tone', 'Not provided')}",
        "",
        f"PRODUCT: {p.get('name', 'Unknown')}",
        f"Product description: {p.get('description', 'Not provided')}",
        "",
        f"CAMPAIGN BRIEF: {state['brief_text']}",
    ]
    if state.get("target_audience"):
        lines.append(f"TARGET AUDIENCE: {state['target_audience']}")
    lines.append("\nGenerate 2-3 creative campaign concepts as a JSON array.")
    return "\n".join(lines)


def _validate_concepts(raw: list) -> list[dict]:
    """
    Validates that each concept dict has the required fields and correct types.
    Raises ValueError with a descriptive message if validation fails.
    """
    required_top = {"name", "description", "visual_dna", "ad_copy", "rationale"}
    required_dna = {"palette", "lighting", "environment", "materials", "mood", "photography_style"}
    required_copy = {"headline", "subcopy", "cta"}

    if not isinstance(raw, list) or len(raw) < 2 or len(raw) > 3:
        raise ValueError(f"Expected list of 2-3 concepts, got: {type(raw).__name__}[{len(raw) if isinstance(raw, list) else '?'}]")

    for i, concept in enumerate(raw):
        missing = required_top - set(concept.keys())
        if missing:
            raise ValueError(f"Concept {i} missing required fields: {missing}")
        missing_dna = required_dna - set(concept["visual_dna"].keys())
        if missing_dna:
            raise ValueError(f"Concept {i} visual_dna missing: {missing_dna}")
        missing_copy = required_copy - set(concept["ad_copy"].keys())
        if missing_copy:
            raise ValueError(f"Concept {i} ad_copy missing: {missing_copy}")

    return raw


def _parse_llm_json(text: str) -> list[dict]:
    """
    Extracts a JSON array from LLM output, stripping any surrounding
    markdown fences or prose the model occasionally emits despite instructions.
    """
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Find the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM output: {text[:200]}")
    return json.loads(text[start:end + 1])


async def _call_gemini(user_prompt: str) -> list[dict]:
    """
    Calls Gemini Flash with structured JSON output requirements.
    Controlled by LLM_PROVIDER=gemini and GEMINI_API_KEY env vars.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is required for LLM_PROVIDER=gemini")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=DIRECTOR_SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.9,
            response_mime_type="application/json",
        ),
    )
    response = await model.generate_content_async(user_prompt)
    return _parse_llm_json(response.text)


async def _call_groq(user_prompt: str) -> list[dict]:
    """
    Calls Groq (llama-3.3-70b-versatile by default) with the Director's
    system prompt. Controlled by LLM_PROVIDER=groq + GROQ_API_KEY env vars;
    GROQ_MODEL overrides the model.

    Uses the official `groq` SDK (OpenAI-compatible chat completions). Groq's
    forced `json_object` response_format only emits top-level OBJECTS, but the
    Director's contract is a top-level ARRAY of concepts — so we rely on the
    system prompt plus the existing _parse_llm_json / _validate_concepts /
    retry-on-malformed path instead of response_format.
    """
    from groq import AsyncGroq

    api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY env var is required for LLM_PROVIDER=groq")

    client = AsyncGroq(api_key=api_key)
    chat_completion = await client.chat.completions.create(
        model=settings.GROQ_MODEL,
        temperature=0.9,
        messages=[
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = chat_completion.choices[0].message.content or ""
    logger.info("_call_groq: model=%s", settings.GROQ_MODEL)
    return _parse_llm_json(text)


# ---------------------------------------------------------------------------
# Graph 1: Director — campaign brief -> 2-3 CreativeConcepts
# ---------------------------------------------------------------------------

class DirectorState(TypedDict):
    brand_context: dict          # {name, description, colors, fonts, tone}
    product_context: dict        # {name, description, image_urls}
    brief_text: str
    target_audience: str | None
    concepts: list[dict]         # populated by generate_concepts node


async def generate_concepts_node(state: DirectorState) -> DirectorState:
    """
    Single LLM call (Groq by default, Gemini Flash as a swappable option)
    producing 2-3 CreativeConcept payloads (name, description, visual_dna,
    ad_copy, rationale) as structured JSON.

    LLM provider is determined by LLM_PROVIDER env var (default: "groq").
    Validates the JSON output against the required schema; retries once on
    malformed output before raising.

    Note: the DB column is named `copy` (frozen schema) but the graph-layer
    and API use `ad_copy` to avoid shadowing Python's built-in / Pydantic's
    BaseModel.copy. The route layer translates ad_copy -> copy before
    persisting to DB.
    """
    llm_provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    user_prompt = _build_director_user_prompt(state)

    last_error: Exception | None = None
    for attempt in range(2):  # 1 initial + 1 retry on malformed output
        try:
            if llm_provider == "groq":
                raw = await _call_groq(user_prompt)
            elif llm_provider == "gemini":
                raw = await _call_gemini(user_prompt)
            else:
                raise RuntimeError(
                    f"Unsupported LLM_PROVIDER: {llm_provider!r}. Supported: 'groq', 'gemini'"
                )

            concepts = _validate_concepts(raw)
            state["concepts"] = concepts
            logger.info("generate_concepts_node: produced %d concepts (attempt %d)", len(concepts), attempt + 1)
            return state

        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("generate_concepts_node: malformed output on attempt %d: %s", attempt + 1, exc)
            if attempt == 0:
                # Amend the prompt on retry to be more explicit about failures
                user_prompt = (
                    user_prompt
                    + f"\n\nPrevious attempt failed validation: {exc}. "
                    "Return ONLY a valid JSON array with no extra text."
                )
            continue

    raise RuntimeError(
        f"generate_concepts_node failed after 2 attempts. Last error: {last_error}"
    )


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

# Composition rules by placement — drives product position and safe areas.
# These are intentionally opinionated defaults; Checkpoint 3+ can make them
# LLM-derived for more interesting per-placement variation.
_PLACEMENT_COMPOSITION = {
    "ig_feed": {
        "framing": "medium shot, product centered, slight left offset",
        "negative_space": "top 20% reserved for headline copy",
        "copy_safe_area": "top 20%",
        "cta_safe_area": "bottom 15%",
        "product_position": "center",
        "product_scale": "large (60% frame height)",
    },
    "ig_story": {
        "framing": "close-up product hero, vertical fill",
        "negative_space": "bottom 25% for copy overlay",
        "copy_safe_area": "bottom 25%",
        "cta_safe_area": "bottom 10%",
        "product_position": "upper center",
        "product_scale": "large (70% frame height)",
    },
    "website_hero": {
        "framing": "wide establishing shot, product right-aligned",
        "negative_space": "left 45% for headline and CTA",
        "copy_safe_area": "left 45%",
        "cta_safe_area": "left 35%, lower third",
        "product_position": "right",
        "product_scale": "medium (50% frame height)",
    },
}


class PlannerState(TypedDict):
    concept: dict          # the selected CreativeConcept, serialized
    brand_context: dict
    product_context: dict
    asset_specs: list[dict]  # populated by plan_assets node


def _build_generation_prompt(
    placement_key: str,
    concept: dict,
    brand_context: dict,
    product_context: dict,
    spec: dict,
) -> str:
    """
    Builds a FLUX/provider-ready generation prompt from the asset spec fields.
    The prompt is DERIVED — it is stored in asset_spec.generation_prompt but
    is not the source of truth. Swapping FLUX for another model only requires
    changing this function.

    ad_copy key is used here because the graph layer uses ad_copy. The DB
    column is named `copy` — the route layer handles the translation.
    """
    comp = _PLACEMENT_COMPOSITION.get(placement_key, {})
    dna = concept.get("visual_dna", {})

    product_name = product_context.get("name", "the product")
    palette = ", ".join(dna.get("palette", []))
    mood = ", ".join(dna.get("mood", []))

    parts = [
        f"Product photography ad for {product_name}.",
        f"Visual style: {dna.get('photography_style', 'editorial')}.",
        f"Environment: {dna.get('environment', 'studio')}.",
        f"Lighting: {dna.get('lighting', 'natural')}.",
        f"Color palette: {palette}." if palette else "",
        f"Mood: {mood}." if mood else "",
        f"Composition: {comp.get('framing', '')}.",
        f"Product placement: {comp.get('product_position', 'center')}, scale {comp.get('product_scale', 'large')}.",
        f"Safe zone for copy text: {comp.get('copy_safe_area', 'top 20%')}.",
        # Copy/headlines are overlaid in post-production (copy_safe_area reserves
        # the space). Asking FLUX to render text at 4 inference steps produces
        # garbled text that fails the OCR legibility check (critical_text_error)
        # and kills the asset — so keep the generated frame text-free.
        "Do not render any text, words, logos, or typography in the image.",
        "Preserve product identity exactly — reference image conditioning required.",
        "Photorealistic, high production value, ready for publication.",
    ]
    return " ".join(p for p in parts if p)


def plan_assets_node(state: PlannerState) -> PlannerState:
    """
    For each of the 3 fixed placements, derive composition rules
    (product_position, product_scale, copy_safe_area, cta_safe_area,
    framing) from the concept's visual_dna, then render a model-specific
    generation_prompt. AssetSpec stays canonical; generation_prompt is
    derived so swapping FLUX later doesn't touch this schema.
    """
    concept = state["concept"]
    brand_context = state["brand_context"]
    product_context = state["product_context"]
    dna = concept.get("visual_dna", {})

    specs = []
    for p in FIXED_PLACEMENTS:
        placement_key = p["placement"]
        comp = _PLACEMENT_COMPOSITION.get(placement_key, {})

        spec = {
            **p,
            "subject": {
                "product_position": comp.get("product_position", "center"),
                "product_scale": comp.get("product_scale", "large"),
                "preserve_identity": True,
            },
            "environment": dna.get("environment"),
            "lighting": dna.get("lighting"),
            "composition": {
                "framing": comp.get("framing"),
                "negative_space": comp.get("negative_space"),
                "copy_safe_area": comp.get("copy_safe_area"),
                "cta_safe_area": comp.get("cta_safe_area"),
            },
            "style": dna.get("mood", []),
            "generation_prompt": None,  # filled below
        }
        spec["generation_prompt"] = _build_generation_prompt(
            placement_key, concept, brand_context, product_context, spec
        )
        specs.append(spec)

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
    prompt_used: str | None              # exact prompt sent (incl. correction addendum)
    infra_error: str | None              # real provider exception (for INFRA_FAILED)
    outcome: Literal["approved", "retry", "manual_review", "infra_failed"] | None


async def generate_image_node(state: GenerationState, provider: ImageGenerationProvider) -> GenerationState:
    prompt = state["asset_spec"]["generation_prompt"]
    if state["corrective_instruction"]:
        prompt = f"{prompt}\n\nCorrection: {state['corrective_instruction']}"
    state["prompt_used"] = prompt
    try:
        state["generated_image_url"] = await provider.generate(
            product_images=state["product_images"],
            reference_images=state["reference_images"],
            prompt=prompt,
            width=state["asset_spec"]["width"],
            height=state["asset_spec"]["height"],
        )
    except Exception as exc:
        # Preserve the real message so INFRA_FAILED rows are actually diagnosable.
        state["outcome"] = "infra_failed"
        state["infra_error"] = str(exc)
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
    """
    Turns the failure_reason (plus any concrete VLM issues from the hybrid
    evaluator) into a corrective instruction for the next attempt. Mock
    evaluations carry no issues, so the mock path is unchanged.
    """
    scores = state.get("scores") or {}
    reason = scores.get("failure_reason") or "unknown"
    instruction = f"Fix: {reason}"
    raw = (scores.get("raw_response") or {}).get("vlm") or {}
    issues = raw.get("issues") or []
    if issues:
        instruction += " Issues: " + "; ".join(str(i) for i in issues[:3])
    state["corrective_instruction"] = instruction
    return state


# Note: build_generation_graph wires generate -> evaluate -> (END | diagnose),
# with attempt-count/3-attempt cap enforced by the FastAPI caller, since that
# cap is a business rule tied to CreativeAsset/GenerationAttempt rows, not
# graph-internal state.
