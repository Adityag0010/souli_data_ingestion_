"""
qdrant_db.py — Qdrant collection management, upsert, and similarity search.
"""

import logging
import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)

from app.models.metadata import EnergyNode

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "souli_knowledge_base")

# Embedding dimension for fastembed "BAAI/bge-small-en-v1.5" = 384
VECTOR_SIZE = 384
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ── Singleton client ──────────────────────────────────────────────────────────
_client: Optional[QdrantClient] = None
_embedder = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
    return _client


def get_embedder():
    """Lazy-load the FastEmbed embedder (downloads model on first use)."""
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding  # type: ignore
            _embedder = TextEmbedding(model_name=EMBED_MODEL)
            logger.info("Loaded FastEmbed model: %s", EMBED_MODEL)
        except ImportError:
            raise ImportError(
                "fastembed is required. Install it with: pip install fastembed"
            )
    return _embedder


def embed_text(text: str) -> list[float]:
    """Embed a single string and return the vector as a list of floats."""
    embedder = get_embedder()
    vectors = list(embedder.embed([text]))  # returns generator
    return [float(v) for v in vectors[0]]


# ── Collection Management ─────────────────────────────────────────────────────
def ensure_collection(
    client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION,
    vector_size: int = VECTOR_SIZE,
) -> None:
    """Create the Qdrant collection if it does not already exist."""
    client = client or get_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s (dim=%d)", collection_name, vector_size)
    else:
        logger.debug("Qdrant collection '%s' already exists.", collection_name)


# ── Upsert ────────────────────────────────────────────────────────────────────
def upsert_nodes(
    nodes: list[EnergyNode],
    collection_name: str = QDRANT_COLLECTION,
    batch_size: int = 64,
) -> int:
    """
    Embed and upsert a list of EnergyNodes into Qdrant.

    Vectorized field: node.embed_text() = "main_question category"
    Stored payload:   node.to_payload() = full Tiered JSON object

    Returns:
        Number of successfully upserted points.
    """
    if not nodes:
        return 0

    client = get_client()
    ensure_collection(client, collection_name)

    points: list[PointStruct] = []
    for node in nodes:
        vector = embed_text(node.embed_text())
        payload = node.to_payload()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    # Batch upsert
    total_upserted = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total_upserted += len(batch)
        logger.info("Upserted batch %d-%d (%d total)", i, i + len(batch), total_upserted)

    return total_upserted


# ── Similarity Search ─────────────────────────────────────────────────────────
def search_nodes(
    query: str,
    k: int = 3,
    collection_name: str = QDRANT_COLLECTION,
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    Perform a similarity search in Qdrant.

    Args:
        query:            Natural language query string.
        k:                Number of results to return.
        collection_name:  Qdrant collection to search.
        score_threshold:  Minimum cosine similarity score (0 = no filter).

    Returns:
        List of payload dicts (each is a full EnergyNode record) sorted by score.
    """
    client = get_client()
    query_vector = embed_text(query)

    results: list[ScoredPoint] = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k,
        score_threshold=score_threshold if score_threshold > 0 else None,
        with_payload=True,
    )

    output = []
    for hit in results:
        record = hit.payload or {}
        record["_score"] = hit.score
        output.append(record)

    logger.info("Search returned %d results for query: %r", len(output), query[:80])
    return output


# ── Collection Stats ──────────────────────────────────────────────────────────
def collection_info(collection_name: str = QDRANT_COLLECTION) -> dict:
    """Return a summary of the collection (point count, vector config)."""
    client = get_client()
    info = client.get_collection(collection_name)
    return {
        "collection": collection_name,
        "points_count": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance.value,
    }
