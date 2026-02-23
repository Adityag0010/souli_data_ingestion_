"""
extractor.py — YouTube transcript processing + Llama 3 multi-row extraction.

Key design goals:
  1. One video → 3-6 EnergyNode rows (distinct coaching use-cases).
  2. Strict few-shot JSON-array prompt to keep Llama 3 on track.
  3. Regex-based JSON extraction handles prose wrapped around the array.
  4. Tenacity retry (3 attempts) for transient LLM failures.
  5. Pydantic validation rejects malformed rows gracefully.
"""

import json
import logging
import os
import re
import sys
from typing import List

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.models.metadata import EnergyNode, Pillars, Atmosphere
from app.services.text_utils import fetch_transcript, clean_transcript, truncate_transcript

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# (Environment variables are now looked up dynamically in get_llm)

# ── Few-Shot Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert coaching analyst for Souli, an emotional wellbeing platform.
Your job is to read a coaching transcript and extract 3 to 6 distinct use-cases or
problem statements that the coach addresses. For each use-case, you output a single
JSON object. Return ONLY a valid JSON array — no explanatory text, no markdown fences.

Each object in the array MUST follow this exact schema:
{
  "main_question": "<The core user struggle addressed — one clear sentence>",
  "category": "<Emotional category, e.g. Anxiety, Burnout, Grief, Self-Worth, Relationships, Purpose>",
  "pillars": {
    "intervention_narrative": "<A short story or metaphor the coach uses to reframe this problem>",
    "intervention_action": "<A concrete exercise or practice the coach recommends for this problem>",
    "intervention_shift": "<One-liner mindset shift the coach is offering>"
  },
  "atmosphere": {
    "tone": "<Tone of the coach for this segment, e.g. 'warm and grounded'>",
    "pacing": "<Pace of delivery, e.g. 'slow and reflective'>"
  },
  "overflow": ["<unique phrase or gem 1>", "<unique phrase or gem 2>"]
}

────── FEW-SHOT EXAMPLES ──────

INPUT (excerpt from a coaching transcript):
"When you feel overwhelmed, your nervous system is not broken — it is working exactly
as designed. The body is saying: slow down, something here needs your attention.
I like to think of anxiety as a smoke alarm. It doesn't mean the house is on fire.
It means: please check the kitchen. The practice I give my clients is a 3-minute
body scan every morning — just noticing, not fixing. Over time this builds a language
between you and your body. The shift is moving from 'what is wrong with me' to
'what is my body trying to tell me.'"

OUTPUT:
[
  {
    "main_question": "How do I stop feeling overwhelmed when my nervous system is in overdrive?",
    "category": "Anxiety",
    "pillars": {
      "intervention_narrative": "Anxiety is like a smoke alarm — it signals 'check the kitchen', not 'the house is on fire'.",
      "intervention_action": "Practice a 3-minute morning body scan — notice sensations without trying to fix them.",
      "intervention_shift": "Move from 'what is wrong with me' to 'what is my body trying to tell me.'"
    },
    "atmosphere": {
      "tone": "warm and reassuring",
      "pacing": "slow and deliberate"
    },
    "overflow": [
      "Your nervous system is not broken — it is working exactly as designed.",
      "Build a language between you and your body."
    ]
  }
]

────── END OF EXAMPLES ──────

Now analyze the following transcript and produce 3 to 6 objects using the same schema.
Return ONLY the JSON array. Do not include any text outside the array.
"""

USER_PROMPT_TEMPLATE = """\
TRANSCRIPT:
{transcript}

Return the JSON array now:
"""


# ── LLM Factory ───────────────────────────────────────────────────────────────
def get_llm():
    """Return the configured LangChain chat model."""
    load_dotenv(override=True)
    logger.info("DEBUG: Extractor logic version 2.0")
    logger.info("DEBUG: os.environ['GROQ_MODEL'] = %s", os.environ.get("GROQ_MODEL"))
    
    llm_type = os.getenv("LLM_TYPE", "ollama").lower()
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    if llm_type == "groq":
        if not groq_api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in .env")
        from langchain_groq import ChatGroq  # type: ignore
        return ChatGroq(api_key=groq_api_key, model=groq_model, temperature=0.3)

    # Default: Ollama (local)
    from langchain_ollama import ChatOllama  # type: ignore
    return ChatOllama(base_url=ollama_base_url, model=ollama_model, temperature=0.3)


# ── JSON extraction helper ────────────────────────────────────────────────────
def _extract_json_array(text: str) -> list:
    """
    Extract the first JSON array from a string.
    Handles cases where the LLM wraps the array in prose or markdown fences.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find outermost [...] block via regex
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON array found in LLM response.")


# ── Core Extraction ───────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
    reraise=True,
)
def _call_llm_and_parse(transcript: str) -> list[dict]:
    """
    Call the LLM and parse the JSON array response.
    Retried up to 3 times on parse failures.
    """
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(transcript=transcript)),
    ]

    llm_type = os.getenv("LLM_TYPE", "ollama").lower()
    model_name = os.getenv("GROQ_MODEL" if llm_type == "groq" else "OLLAMA_MODEL", "unknown")
    
    logger.info("Calling LLM (%s/%s) ...", llm_type, model_name)
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    logger.debug("Raw LLM response (first 500 chars): %s", raw[:500])

    return _extract_json_array(raw)


def _validate_nodes(raw_list: list, video_id: str, video_url: str) -> list[EnergyNode]:
    """
    Validate raw dicts against the EnergyNode schema.
    Invalid items are logged and skipped rather than crashing the pipeline.
    """
    nodes: list[EnergyNode] = []
    for i, item in enumerate(raw_list):
        try:
            node = EnergyNode(
                video_id=video_id,
                video_url=video_url,
                main_question=item["main_question"],
                category=item["category"],
                pillars=Pillars(**item["pillars"]),
                atmosphere=Atmosphere(**item["atmosphere"]),
                overflow=item.get("overflow", []),
            )
            nodes.append(node)
        except Exception as exc:
            logger.warning("Skipping invalid node %d: %s — %s", i, exc, item)

    return nodes


def extract_energy_nodes(
    transcript: str,
    video_id: str,
    video_url: str,
    max_chars: int = 12_000,
) -> list[EnergyNode]:
    """
    Extract 3–6 EnergyNodes from a coaching transcript.

    Args:
        transcript:  Cleaned transcript text.
        video_id:    YouTube video ID.
        video_url:   Full YouTube URL.
        max_chars:   Maximum transcript chars sent to LLM.

    Returns:
        List of validated EnergyNode objects (may be empty on total failure).
    """
    truncated = truncate_transcript(transcript, max_chars=max_chars)
    logger.info("Transcript length: %d chars (original), %d chars (truncated)", len(transcript), len(truncated))

    try:
        raw_list = _call_llm_and_parse(truncated)
    except Exception as exc:
        logger.error("LLM extraction failed after retries for %s: %s", video_id, exc)
        return []

    nodes = _validate_nodes(raw_list, video_id, video_url)
    logger.info("Extracted %d valid EnergyNodes for video_id=%s", len(nodes), video_id)
    return nodes


# ── End-to-End Pipeline for One URL ──────────────────────────────────────────
def process_youtube_url(yt_url: str) -> tuple[str, list[EnergyNode]]:
    """
    Full pipeline: URL → transcript → clean → extract → validate.

    Returns:
        (video_id, list_of_energy_nodes)
    """
    video_id, raw_transcript = fetch_transcript(yt_url)
    cleaned = clean_transcript(raw_transcript)
    nodes = extract_energy_nodes(cleaned, video_id, yt_url)
    return video_id, nodes
