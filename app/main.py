"""
main.py — FastAPI entry point for the Souli Data Ingestion & Retrieval System.

Endpoints:
  GET  /health           — health check
  POST /process-csv      — upload CSV of YouTube links, extract & upsert to Qdrant
  POST /query            — natural language similarity search
  GET  /collection-info  — Qdrant collection stats
"""

import io
import logging
import os
import tempfile
from typing import List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.services.extractor import process_youtube_url
from app.services.qdrant_db import upsert_nodes, search_nodes, collection_info, ensure_collection

load_dotenv(override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Souli AI — Data Ingestion & Retrieval API",
    description=(
        "Souli AI Mobile Application — an AI-powered emotional wellness companion "
        "that supports users through daily emotional challenges using safe emotional "
        "expression, personalized insights, and short guided practices. "
        "This pipeline extracts structured EnergyNodes (with DiagnosticLayer routing) "
        "from YouTube coaching transcripts and stores them in Qdrant."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    k: int = 3
    score_threshold: float = 0.0


class QueryResponse(BaseModel):
    results: List[dict]
    count: int


class ProcessCSVResponse(BaseModel):
    processed_links: int
    total_nodes_extracted: int
    total_nodes_upserted: int
    failed_links: List[str]
    preview: List[dict]


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Souli API starting up — ensuring Qdrant collection exists ...")
    try:
        ensure_collection()
    except Exception as exc:
        logger.warning("Could not connect to Qdrant at startup: %s", exc)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "souli-ingestion-api"}


# ── Collection Info ───────────────────────────────────────────────────────────
@app.get("/collection-info", tags=["System"])
def get_collection_info():
    try:
        return collection_info()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}")


# ── Process CSV ───────────────────────────────────────────────────────────────
@app.post("/process-csv", response_model=ProcessCSVResponse, tags=["Ingestion"])
async def process_csv(file: UploadFile = File(...)):
    """
    Accept a CSV file with at least one column: `yt_link`.
    For each link:
      1. Fetch & clean the YouTube transcript.
      2. Run Llama 3 multi-row extraction → 3–6 EnergyNodes.
      3. Upsert nodes to Qdrant.
    Returns a preview of extracted rows and a download link for the full CSV.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "yt_link" not in df.columns:
        # Try common alternatives
        alt_cols = [c for c in df.columns if "link" in c or "url" in c or "youtube" in c]
        if not alt_cols:
            raise HTTPException(
                status_code=422,
                detail=f"CSV must contain a 'yt_link' column. Found: {list(df.columns)}",
            )
        df = df.rename(columns={alt_cols[0]: "yt_link"})

    links = df["yt_link"].dropna().unique().tolist()
    logger.info("Processing %d unique YouTube links from CSV", len(links))

    all_nodes = []
    failed_links: list[str] = []

    for url in links:
        try:
            video_id, nodes = process_youtube_url(str(url).strip())
            logger.info("Extracted %d nodes from %s", len(nodes), url)
            all_nodes.extend(nodes)
        except Exception as exc:
            logger.error("Failed to process %s: %s", url, exc)
            failed_links.append(str(url))

    # Upsert to Qdrant
    total_upserted = 0
    if all_nodes:
        try:
            total_upserted = upsert_nodes(all_nodes)
        except Exception as exc:
            logger.error("Qdrant upsert failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Qdrant upsert failed: {exc}")

    # Build flat records for preview & CSV export
    records = [node.to_payload() for node in all_nodes]
    if records:
        export_df = pd.json_normalize(records)
    else:
        # Create empty DataFrame with expected columns (including diagnostic_layer)
        export_df = pd.DataFrame(columns=[
            "video_id", "video_url", "main_question", "category",
            # Diagnostic Layer — deciding factor for energy-node routing
            "diagnostic_layer.related_inner_issues",
            "diagnostic_layer.reality_commitment_check",
            "diagnostic_layer.hidden_benefit",
            "diagnostic_layer.energy_node",
            # Response pillars — how to talk after knowing the problem
            "pillars.intervention_narrative", "pillars.intervention_action", "pillars.intervention_shift",
            "atmosphere.tone", "atmosphere.pacing",
            "overflow",
        ])

    # Save to data/ directory
    os.makedirs("data", exist_ok=True)
    export_path = "data/extracted_nodes.csv"
    export_df.to_csv(export_path, index=False)
    logger.info("Exported extracted nodes to %s (%d rows)", export_path, len(export_df))

    return ProcessCSVResponse(
        processed_links=len(links) - len(failed_links),
        total_nodes_extracted=len(all_nodes),
        total_nodes_upserted=total_upserted,
        failed_links=failed_links,
        preview=records[:10],  # Return first 10 rows in the response
    )


@app.get("/download-csv", tags=["Ingestion"])
def download_csv():
    """Download the last extracted nodes CSV."""
    path = "data/extracted_nodes.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No extracted CSV found. Run /process-csv first.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    except Exception as exc:
        logger.error("Could not read CSV file: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error reading CSV: {exc}")

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=souli_extracted_nodes.csv"},
    )


# ── Query ─────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
def query(request: QueryRequest):
    """
    Perform a semantic similarity search over the Qdrant knowledge base.
    Returns k matching EnergyNodes with their Pillars, Atmosphere, and Overflow.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = search_nodes(
            query=request.query,
            k=request.k,
            score_threshold=request.score_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}")

    return QueryResponse(results=results, count=len(results))
