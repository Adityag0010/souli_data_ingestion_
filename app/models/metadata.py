"""
Pydantic models for the Souli Tiered Metadata Architecture.

EnergyNode is the top-level record that gets stored in Qdrant.
Each coaching video produces 3–6 EnergyNodes, one per distinct use-case.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Tier 1: Pillars ───────────────────────────────────────────────────────────
class Pillars(BaseModel):
    """The three core pillars of a coaching intervention."""

    intervention_narrative: str = Field(
        ...,
        description="A short story or metaphor the coach uses to reframe the problem.",
    )
    intervention_action: str = Field(
        ...,
        description="A concrete exercise or practice the coach recommends.",
    )
    intervention_shift: str = Field(
        ...,
        description="A one-liner that captures the mindset shift being offered.",
    )


# ── Tier 2: Atmosphere ────────────────────────────────────────────────────────
class Atmosphere(BaseModel):
    """The emotional atmosphere and pacing the coach creates."""

    tone: str = Field(
        ...,
        description="Descriptive tone of the coach (e.g. 'warm and grounded', 'playful and curious').",
    )
    pacing: str = Field(
        ...,
        description="Pace of coaching delivery (e.g. 'slow and reflective', 'brisk and energizing').",
    )


# ── Top-Level: EnergyNode ──────────────────────────────────────────────────────
class EnergyNode(BaseModel):
    """
    A single diagnostic row representing one distinct coaching use-case
    extracted from a YouTube video transcript.
    """

    video_id: str = Field(..., description="YouTube video ID.")
    video_url: str = Field(..., description="Full YouTube URL.")
    main_question: str = Field(
        ...,
        description="The core user struggle or question this node addresses.",
    )
    category: str = Field(
        ...,
        description="High-level emotional category (e.g. 'Anxiety', 'Burnout', 'Relationships').",
    )
    pillars: Pillars
    atmosphere: Atmosphere
    overflow: List[str] = Field(
        default_factory=list,
        description="Catch-all list of unique phrases, gems, or insights that don't fit the pillars.",
    )

    def embed_text(self) -> str:
        """Returns the text to be vectorized for similarity search."""
        return f"{self.main_question} {self.category}"

    def to_payload(self) -> dict:
        """Returns the full JSON payload to be stored in Qdrant."""
        return self.model_dump()
