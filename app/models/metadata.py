"""
metadata.py — Pydantic models for the Souli AI Mobile Application.

Souli is an AI-powered emotional wellness companion that supports users
through daily emotional challenges using safe emotional expression,
personalized insights, and short guided practices.

EnergyNode is the top-level record stored in Qdrant.
Each coaching video produces 3–6 EnergyNodes, one per distinct use-case.

Architecture:
  ┌─ EnergyNode ──────────────────────────────────────┐
  │  video_id / video_url      ← provenance            │
  │  main_question             ← user surface struggle  │
  │  category                  ← broad emotional label  │
  │                                                     │
  │  diagnostic_layer ← DECIDING FACTOR FOR ROUTING    │
  │    related_inner_issues    ← root psychological causes │
  │    reality_commitment_check← user readiness test    │
  │    hidden_benefit          ← secondary gain for staying stuck │
  │    energy_node             ← canonical block label  │
  │                                                     │
  │  pillars  ← HOW TO TALK after knowing the problem  │
  │    intervention_narrative  ← reframing story/metaphor │
  │    intervention_action     ← concrete practice     │
  │    intervention_shift      ← mindset shift one-liner │
  │                                                     │
  │  atmosphere ← coaching delivery style              │
  │    tone / pacing                                    │
  │                                                     │
  │  overflow  ← catch-all gems / phrases              │
  └────────────────────────────────────────────────────┘

Embedding vector: main_question + category + full diagnostic_layer
(all four diagnostic fields are concatenated so the RAG query lands
on the right energy block, not just the surface label.)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Tier 0: Diagnostic Layer ──────────────────────────────────────────────────
class DiagnosticLayer(BaseModel):
    """
    The deciding factor for energy-node routing.

    These four fields are derived from the Souli Energy Framework
    (Souli_EnergyFramework_PW.xlsx — ExpressionsMapping sheet) and tell the
    RAG system *which* energy block a user is sitting in before any coaching
    response is composed.

    All four fields are mandatory — if the LLM cannot determine a value it
    must make its best inference from the coaching transcript context.
    """

    related_inner_issues: str = Field(
        ...,
        description=(
            "The deep psychological and emotional root causes that typically drive "
            "the surface question. These are the inner dynamics a mindfulness coach "
            "would recognize underneath the presenting problem — such as emotional "
            "labour, suppressed needs, fear of abandonment, perfectionism, or "
            "unprocessed grief. List the most relevant ones, comma-separated or as "
            "a short phrase. Example: 'emotional labour, over-caring, burnout, "
            "suppressed needs'."
        ),
    )

    reality_commitment_check: str = Field(
        ...,
        description=(
            "A single, direct yes/no question that exposes whether the user is "
            "genuinely ready to change their situation. This question cuts through "
            "ambivalence and reveals the user's actual commitment level. It should "
            "be phrased in second person, present tense, and reference the core "
            "desired shift. Example: 'Do you want to feel restful?' or 'Are you "
            "willing to let go of the need to control outcomes?'"
        ),
    )

    hidden_benefit: str = Field(
        ...,
        description=(
            "The unconscious psychological or emotional pay-off the user receives "
            "from staying stuck in the current pattern. This is the 'secondary gain' "
            "— the hidden reason the pattern persists despite causing suffering. "
            "Understanding this is critical for designing the right intervention. "
            "Examples: 'avoiding painful decisions', 'maintaining a victim identity "
            "that generates sympathy', 'keeping others at a distance to avoid "
            "vulnerability'. Should be a short phrase or sentence."
        ),
    )

    energy_node: str = Field(
        ...,
        description=(
            "The canonical energy-block label from the Souli Inner Energy Framework. "
            "This is the primary routing key used to match users to specific "
            "meditation programs and healing practices. Must be a snake_case label "
            "that reflects the dominant energetic pattern. Common values include: "
            "'blocked_energy' (suppression, numbness, stuck grief), "
            "'outofcontrol_energy' (overwhelm, rage, anxiety spiral), "
            "'scattered_energy' (unfocused, restless, distracted mind), "
            "'depleted_energy' (burnout, exhaustion, chronic fatigue), "
            "'collapsed_energy' (depression, hopelessness, withdrawal), "
            "'hypervigilant_energy' (anxiety, over-monitoring, fear-based living), "
            "'disconnected_energy' (dissociation, numbness, identity confusion), "
            "'wounded_energy' (unresolved trauma, deep grief, abandonment wounds). "
            "Choose the single most fitting label or coin a new snake_case label if "
            "none of the above match."
        ),
    )


# ── Tier 1: Pillars ───────────────────────────────────────────────────────────
class Pillars(BaseModel):
    """
    The three core pillars of a coaching intervention.

    These define HOW the coach talks to the user after the energy node has
    been identified. They shape the tone, metaphor, and practice recommendation.
    """

    intervention_narrative: str = Field(
        ...,
        description=(
            "A short story, metaphor, or reframe the coach uses to shift the user's "
            "perspective on their problem. This makes the issue feel less threatening "
            "and more workable. Example: 'Anxiety is like a smoke alarm — it does not "
            "mean the house is on fire, it means: please check the kitchen.'"
        ),
    )
    intervention_action: str = Field(
        ...,
        description=(
            "A concrete, time-bounded exercise or mindfulness practice the coach "
            "recommends for this specific problem. Should be specific enough that a "
            "user can do it immediately. Example: 'Practice a 3-minute morning body "
            "scan — notice sensations without trying to fix them.'"
        ),
    )
    intervention_shift: str = Field(
        ...,
        description=(
            "A one-liner that captures the core mindset shift being offered. Usually "
            "structured as a reframe from an old belief to a new perspective. "
            "Example: 'Move from what is wrong with me to what is my body trying "
            "to tell me.'"
        ),
    )


# ── Tier 2: Atmosphere ────────────────────────────────────────────────────────
class Atmosphere(BaseModel):
    """The emotional atmosphere and pacing the coach creates during delivery."""

    tone: str = Field(
        ...,
        description=(
            "Descriptive tone of the coach in this coaching segment. Use two to three "
            "adjectives that capture the emotional quality of delivery. "
            "Examples: 'warm and grounded', 'playful and curious', 'firm and compassionate', "
            "'gentle and unhurried'."
        ),
    )
    pacing: str = Field(
        ...,
        description=(
            "The pace and rhythm of the coaching delivery for this segment. "
            "Examples: 'slow and reflective', 'brisk and energizing', "
            "'measured and deliberate', 'flowing and conversational'."
        ),
    )


# ── Top-Level: EnergyNode ──────────────────────────────────────────────────────
class EnergyNode(BaseModel):
    """
    A single diagnostic row representing one distinct coaching use-case
    extracted from a YouTube video transcript.

    Stored in Qdrant as a vector point.
    The embedding vector is computed from: main_question + category +
    the full diagnostic_layer (all four fields) so that semantic search
    lands on the correct energy block, not just the surface topic.
    """

    video_id: str = Field(..., description="YouTube video ID (extracted from URL).")
    video_url: str = Field(..., description="Full YouTube video URL.")
    main_question: str = Field(
        ...,
        description=(
            "The core user struggle or question this node addresses, expressed as a "
            "single clear sentence in the first or second person. This is the surface "
            "expression of the problem — what the user would actually say or search. "
            "Example: 'How do I stop feeling overwhelmed when my nervous system is "
            "in overdrive?'"
        ),
    )
    category: str = Field(
        ...,
        description=(
            "High-level emotional category that groups this node. Must be one of: "
            "Anxiety, Burnout, Grief, Self-Worth, Relationships, Purpose, Anger, "
            "Emotional Health, Relationship Awareness, Trauma, Identity, Loneliness, "
            "Fear, Shame, or another relevant label if none fit."
        ),
    )

    # ── Diagnostic Layer — deciding factor for energy-node routing ────────────
    diagnostic_layer: DiagnosticLayer = Field(
        ...,
        description=(
            "The four-field diagnostic block that determines which energy node "
            "the user is sitting in. Used as the primary deciding factor for "
            "routing to the correct healing program in the Souli framework. "
            "All four sub-fields are mandatory."
        ),
    )

    # ── Response Layer — how to talk after knowing the problem ────────────────
    pillars: Pillars = Field(
        ...,
        description=(
            "The three coaching response pillars: narrative reframe, concrete action, "
            "and mindset shift. These define how Souli AI talks to the user once the "
            "energy node has been identified."
        ),
    )
    atmosphere: Atmosphere = Field(
        ...,
        description="The emotional tone and pacing of the coaching delivery.",
    )
    overflow: List[str] = Field(
        default_factory=list,
        description=(
            "Catch-all list of unique phrases, coaching gems, or key insights from "
            "the transcript that do not fit neatly into the pillars but are too "
            "valuable to discard. 2–5 short phrases or sentences."
        ),
    )

    def embed_text(self) -> str:
        """
        Returns the text to be vectorized for similarity search.

        Uses main_question + category + the full diagnostic_layer as the
        deciding factor, ensuring the vector space captures both the surface
        struggle and the deep psychological root (energy block) together.
        """
        dl = self.diagnostic_layer
        return (
            f"{self.main_question} "
            f"{self.category} "
            f"{dl.related_inner_issues} "
            f"{dl.reality_commitment_check} "
            f"{dl.hidden_benefit} "
            f"{dl.energy_node}"
        )

    def to_payload(self) -> dict:
        """Returns the full JSON payload to be stored in Qdrant."""
        return self.model_dump()
