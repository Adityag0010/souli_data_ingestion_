"""
ui/app.py — Streamlit Admin Dashboard for Souli.

Tab 1: Ingestion
  - Upload a CSV with YouTube links
  - Shows progress while processing
  - Displays a preview table of extracted EnergyNodes
  - Download the full CSV export

Tab 2: Chat Sandbox
  - Enter a coaching struggle / question
  - Retrieves matching EnergyNodes from Qdrant
  - Displays Story, Action, and Vibe cards
"""

import os
import io
import json
import time

import httpx
import pandas as pd
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Souli — Admin Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .souli-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .souli-sub {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .card {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .card-title {
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #667eea;
        margin-bottom: 0.4rem;
    }
    .card-body {
        font-size: 1rem;
        color: #1F2937;
        line-height: 1.7;
    }
    .overflow-tag {
        display: inline-block;
        background: #EDE9FE;
        color: #5B21B6;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.82rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }
    .score-badge {
        background: #D1FAE5;
        color: #065F46;
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    hr { border-color: #E5E7EB; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="souli-header">🌊 Souli Admin Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="souli-sub">Coach Intelligence Pipeline — Ingest · Extract · Retrieve</div>',
    unsafe_allow_html=True,
)

# ── API Health Check ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            st.success("API: Online ✅")
        else:
            st.error("API: Unexpected response")
    except Exception:
        st.error("API: Offline ❌  \nStart FastAPI with:  \n`uvicorn app.main:app --reload`")

    try:
        r2 = httpx.get(f"{API_BASE}/collection-info", timeout=3)
        if r2.status_code == 200:
            info = r2.json()
            st.info(f"Qdrant: **{info.get('points_count', 0)}** points in `{info.get('collection','')}`")
        else:
            st.warning("Qdrant: No collection yet")
    except Exception:
        st.warning("Qdrant: Unreachable")

    st.markdown("---")
    st.markdown("**API Base URL**")
    st.code(API_BASE)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📥 Ingestion", "💬 Chat Sandbox"])


# ════════════════════════════════════════════════════════════════════════════════
# Tab 1: Ingestion
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload a CSV of YouTube Coaching Links")
    st.markdown(
        "The CSV must have at least one column named **`yt_link`** (or `url` / `link`). "
        "One row per YouTube video."
    )

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    with col_sample:
        sample_csv = "yt_link\nhttps://www.youtube.com/watch?v=dQw4w9WgXcQ\nhttps://www.youtube.com/watch?v=example2\n"
        st.download_button(
            label="⬇ Download Sample CSV",
            data=sample_csv,
            file_name="sample_links.csv",
            mime="text/csv",
        )

    if uploaded_file:
        df_preview = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        st.markdown(f"**Detected {len(df_preview)} rows.** Preview:")
        st.dataframe(df_preview.head(5), use_container_width=True)

        if st.button("🚀 Start Extraction & Ingestion", type="primary"):
            progress_bar = st.progress(0, text="Uploading CSV …")

            with st.spinner("Processing YouTube links with Llama 3 — this may take a few minutes …"):
                try:
                    progress_bar.progress(15, text="Sending to API …")
                    uploaded_file.seek(0)
                    response = httpx.post(
                        f"{API_BASE}/process-csv",
                        files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                        timeout=600,  # 10 min for large batches
                    )
                    progress_bar.progress(80, text="Processing complete, loading results …")

                    if response.status_code == 200:
                        data = response.json()
                        progress_bar.progress(100, text="Done!")
                        time.sleep(0.5)
                        progress_bar.empty()

                        # ── Summary Metrics ───────────────────────────────
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Links Processed", data["processed_links"])
                        m2.metric("Nodes Extracted", data["total_nodes_extracted"])
                        m3.metric("Upserted to Qdrant", data["total_nodes_upserted"])
                        m4.metric("Failed Links", len(data["failed_links"]))

                        if data["failed_links"]:
                            with st.expander("⚠️ Failed Links"):
                                for link in data["failed_links"]:
                                    st.code(link)

                        # ── Preview Table ─────────────────────────────────
                        if data["preview"]:
                            st.markdown("#### 🔍 Extracted Nodes Preview (first 10)")
                            preview_df = pd.json_normalize(data["preview"])
                            st.dataframe(preview_df, use_container_width=True)

                        # ── Download CSV ──────────────────────────────────
                        try:
                            csv_resp = httpx.get(f"{API_BASE}/download-csv", timeout=30)
                            if csv_resp.status_code == 200:
                                st.download_button(
                                    label="⬇ Download Full Extracted CSV",
                                    data=csv_resp.content,
                                    file_name="souli_extracted_nodes.csv",
                                    mime="text/csv",
                                )
                        except Exception:
                            st.info("CSV download unavailable — check the API server.")

                    else:
                        progress_bar.empty()
                        st.error(f"API Error {response.status_code}: {response.text}")

                except httpx.ConnectError:
                    progress_bar.empty()
                    st.error("Cannot reach the FastAPI server. Is it running?  \n`uvicorn app.main:app --reload`")
                except Exception as exc:
                    progress_bar.empty()
                    st.error(f"Unexpected error: {exc}")


# ════════════════════════════════════════════════════════════════════════════════
# Tab 2: Chat Sandbox
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 💬 Ask the Coaching Soul")
    st.markdown(
        "Describe a user struggle or emotional challenge. "
        "The system will retrieve the most relevant coaching insights."
    )

    query_input = st.text_area(
        "What's the user struggling with?",
        placeholder="e.g. I feel exhausted and like nothing I do is ever enough …",
        height=110,
    )
    col_k, col_btn = st.columns([1, 2])
    with col_k:
        k_results = st.slider("Results to retrieve", min_value=1, max_value=10, value=3)
    with col_btn:
        search_clicked = st.button("🔍 Find Coaching Insights", type="primary", use_container_width=True)

    if search_clicked:
        if not query_input.strip():
            st.warning("Please enter a struggle or question first.")
        else:
            with st.spinner("Searching knowledge base …"):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/query",
                        json={"query": query_input, "k": k_results},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])

                        if not results:
                            st.info("No matching nodes found. Try ingesting more videos first.")
                        else:
                            st.markdown(f"#### Found **{len(results)}** coaching insight(s)")
                            for i, node in enumerate(results):
                                score = node.get("_score", 0)
                                with st.expander(
                                    f"🌀 Node {i+1} — **{node.get('main_question', 'N/A')}**  "
                                    f"  `{node.get('category', '')}` "
                                    f"  <span class='score-badge'>Score: {score:.3f}</span>",
                                    expanded=(i == 0),
                                ):
                                    pillars = node.get("pillars", {})
                                    atmosphere = node.get("atmosphere", {})
                                    overflow = node.get("overflow", [])

                                    # Story
                                    st.markdown(
                                        f"""<div class="card">
                                        <div class="card-title">📖 The Story (Narrative)</div>
                                        <div class="card-body">{pillars.get('intervention_narrative', '—')}</div>
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )
                                    # Action
                                    st.markdown(
                                        f"""<div class="card">
                                        <div class="card-title">🎯 The Action (Exercise)</div>
                                        <div class="card-body">{pillars.get('intervention_action', '—')}</div>
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )
                                    # Shift
                                    st.markdown(
                                        f"""<div class="card">
                                        <div class="card-title">✨ The Shift (One-liner)</div>
                                        <div class="card-body">{pillars.get('intervention_shift', '—')}</div>
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )
                                    # Atmosphere
                                    col_tone, col_pace = st.columns(2)
                                    col_tone.markdown(f"**🎭 Tone:** {atmosphere.get('tone', '—')}")
                                    col_pace.markdown(f"**⏱ Pacing:** {atmosphere.get('pacing', '—')}")

                                    # Overflow gems
                                    if overflow:
                                        st.markdown("**💎 Gems & Notable Phrases:**")
                                        tags_html = " ".join(
                                            f'<span class="overflow-tag">{phrase}</span>'
                                            for phrase in overflow
                                        )
                                        st.markdown(tags_html, unsafe_allow_html=True)

                                    # Source
                                    st.markdown(
                                        f"<small>🔗 Source: [{node.get('video_url', '')}]({node.get('video_url', '')})</small>",
                                        unsafe_allow_html=True,
                                    )
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")

                except httpx.ConnectError:
                    st.error("Cannot reach the FastAPI server.  \n`uvicorn app.main:app --reload`")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
