"""
tests/test_extractor_smoke.py — Smoke test for the LLM extraction pipeline.

Tests the extractor with a hardcoded transcript snippet (no real YouTube call).
Requires either:
  - Ollama running with llama3 pulled   (LLM_TYPE=ollama, default)
  - GROQ_API_KEY set in .env            (LLM_TYPE=groq)
"""

import pytest
from app.services.extractor import extract_energy_nodes
from app.models.metadata import EnergyNode, DiagnosticLayer

# A realistic coaching transcript excerpt covering three distinct use-cases
SAMPLE_TRANSCRIPT = """
When you feel overwhelmed, your nervous system is not broken — it is working exactly as designed.
The body is saying: slow down, something here needs your attention.
I like to think of anxiety as a smoke alarm. It doesn't mean the house is on fire.
It means: please check the kitchen. The practice I give my clients is a 3-minute body scan 
every morning — just noticing, not fixing. Over time this builds a language between you and your body.
The shift is moving from what is wrong with me to what is my body trying to tell me.

Now let's talk about burnout. So many of my clients come to me completely depleted — not because 
they don't love what they do, but because they have confused motion with meaning. They're busy 
all the time but feel empty. The reframe I offer is this: rest is not a reward for finishing work.
Rest is the foundation from which good work grows. I ask them to schedule one guilt-free rest 
period every day, even fifteen minutes. And the mindset shift is: I am productive because I rest,
not despite resting.

Grief is another area where people get stuck. We tend to think of grief as something linear —
stages you move through and eventually complete. But grief is circular. It comes back. 
Not because you haven't healed, but because love doesn't disappear. The action here is a 
grief journal — writing letters to what or who you have lost, not to get closure, but to stay 
in relationship with that love. The shift: grief is not the opposite of healing — it is healing.
"""

VIDEO_ID = "test123"
VIDEO_URL = "https://youtube.com/watch?v=test123"


def test_extract_returns_list():
    """Extraction should return a list."""
    nodes = extract_energy_nodes(SAMPLE_TRANSCRIPT, video_id=VIDEO_ID, video_url=VIDEO_URL)
    assert isinstance(nodes, list), "extract_energy_nodes should return a list"


def test_extract_minimum_nodes():
    """Should extract at least 2 nodes from a multi-topic transcript."""
    nodes = extract_energy_nodes(SAMPLE_TRANSCRIPT, video_id=VIDEO_ID, video_url=VIDEO_URL)
    assert len(nodes) >= 2, f"Expected at least 2 nodes, got {len(nodes)}"


def test_each_node_is_energy_node():
    """Every item in the list should be a valid EnergyNode."""
    nodes = extract_energy_nodes(SAMPLE_TRANSCRIPT, video_id=VIDEO_ID, video_url=VIDEO_URL)
    for node in nodes:
        assert isinstance(node, EnergyNode), f"Expected EnergyNode, got {type(node)}"


def test_node_core_fields_non_empty():
    """Core fields of each node should be non-empty strings."""
    nodes = extract_energy_nodes(SAMPLE_TRANSCRIPT, video_id=VIDEO_ID, video_url=VIDEO_URL)
    for node in nodes:
        assert node.main_question.strip(), "main_question should not be empty"
        assert node.category.strip(), "category should not be empty"
        assert node.pillars.intervention_narrative.strip(), "narrative should not be empty"
        assert node.pillars.intervention_action.strip(), "action should not be empty"
        assert node.pillars.intervention_shift.strip(), "shift should not be empty"
        assert node.atmosphere.tone.strip(), "tone should not be empty"
        assert node.atmosphere.pacing.strip(), "pacing should not be empty"


def test_diagnostic_layer_fields_non_empty():
    """All four diagnostic_layer fields must be non-empty strings — they are the deciding factor."""
    nodes = extract_energy_nodes(SAMPLE_TRANSCRIPT, video_id=VIDEO_ID, video_url=VIDEO_URL)
    assert len(nodes) > 0, "Need at least one node to test diagnostic_layer"
    for node in nodes:
        dl = node.diagnostic_layer
        assert isinstance(dl, DiagnosticLayer), "diagnostic_layer must be a DiagnosticLayer instance"
        assert dl.related_inner_issues.strip(), "related_inner_issues should not be empty"
        assert dl.reality_commitment_check.strip(), "reality_commitment_check should not be empty"
        assert dl.hidden_benefit.strip(), "hidden_benefit should not be empty"
        assert dl.energy_node.strip(), "energy_node should not be empty"


def test_embed_text_contains_diagnostic_layer():
    """embed_text() must include all diagnostic_layer fields (the deciding factor for routing)."""
    nodes = extract_energy_nodes(SAMPLE_TRANSCRIPT, video_id=VIDEO_ID, video_url=VIDEO_URL)
    assert len(nodes) > 0, "Need at least one node"
    for node in nodes:
        text = node.embed_text()
        assert node.main_question in text, "embed_text must contain main_question"
        assert node.category in text, "embed_text must contain category"
        assert node.diagnostic_layer.energy_node in text, "embed_text must contain energy_node"
        assert node.diagnostic_layer.related_inner_issues in text, "embed_text must contain related_inner_issues"


def test_video_metadata_preserved():
    """video_id and video_url should be passed through correctly."""
    nodes = extract_energy_nodes(SAMPLE_TRANSCRIPT, video_id=VIDEO_ID, video_url=VIDEO_URL)
    for node in nodes:
        assert node.video_id == VIDEO_ID
        assert node.video_url == VIDEO_URL
