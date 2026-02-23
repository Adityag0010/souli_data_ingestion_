"""
text_utils.py — YouTube transcript fetching and cleaning utilities.
"""

import re
import logging
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from various URL formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://youtube.com/shorts/VIDEO_ID
    """
    parsed = urlparse(url.strip())

    if parsed.netloc in ("youtu.be",):
        # https://youtu.be/VIDEO_ID
        return parsed.path.lstrip("/")

    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # Shorts / embed
        path_parts = parsed.path.split("/")
        for i, part in enumerate(path_parts):
            if part in ("shorts", "embed", "v") and i + 1 < len(path_parts):
                return path_parts[i + 1]

    raise ValueError(f"Cannot extract video ID from URL: {url!r}")


def fetch_transcript(yt_url: str, languages: list[str] | None = None) -> tuple[str, str]:
    """
    Fetch and concatenate the transcript for a YouTube video.

    Returns:
        (video_id, raw_transcript_text)

    Raises:
        RuntimeError if transcript cannot be retrieved.
    """
    if languages is None:
        languages = ["en", "en-US", "en-GB"]

    video_id = extract_video_id(yt_url)
    logger.info("Fetching transcript for video_id=%s", video_id)

    api = YouTubeTranscriptApi()

    try:
        segments = api.fetch(video_id, languages=languages)
        text = " ".join(seg.text for seg in segments)
        return video_id, text
    except NoTranscriptFound:
        # Fallback: try auto-generated transcript
        try:
            transcript_list = api.list(video_id)
            transcript = transcript_list.find_generated_transcript(["en"])
            segments = transcript.fetch()
            text = " ".join(seg.text for seg in segments)
            return video_id, text
        except Exception as inner_exc:
            raise RuntimeError(
                f"No transcript found for {video_id}: {inner_exc}"
            ) from inner_exc
    except TranscriptsDisabled as exc:
        raise RuntimeError(f"Transcripts disabled for {video_id}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch transcript for {video_id}: {exc}") from exc


def clean_transcript(raw: str) -> str:
    """
    Clean a raw transcript string:
      - Remove bracketed noise like [Music], [Applause], [00:01:23]
      - Remove speaker labels like "Host:", "Speaker 1:"
      - Collapse excessive whitespace
      - Strip common marketing phrases
    """
    text = raw

    # Remove bracketed content (timestamps, sound effects)
    text = re.sub(r"\[.*?\]", " ", text)

    # Remove speaker labels (e.g. "John:", "Host:", "Speaker 2:")
    text = re.sub(r"^[A-Za-z][A-Za-z0-9 _]{0,30}:\s*", "", text, flags=re.MULTILINE)

    # Remove timestamp patterns (HH:MM:SS or MM:SS)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)

    # Remove common marketing filler
    filler_patterns = [
        r"subscribe\s+to\s+my\s+channel",
        r"like\s+and\s+subscribe",
        r"hit\s+the\s+notification\s+bell",
        r"follow\s+me\s+on\s+(instagram|twitter|facebook|tiktok)",
        r"check\s+out\s+my\s+(website|link\s+in\s+bio)",
        r"use\s+code\s+\w+\s+for\s+\d+%?\s+off",
        r"sponsored\s+by",
        r"this\s+video\s+is\s+brought\s+to\s+you\s+by",
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_transcript(text: str, max_chars: int = 12_000) -> str:
    """
    Truncate a transcript to a maximum character length to stay within
    LLM context windows. Truncation is done at the nearest sentence boundary.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Try to end at a sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        truncated = truncated[: last_period + 1]

    return truncated.strip()
