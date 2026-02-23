# Technical Specification: Souli Coaching Ingestion & Retrieval System

## 1. Project Overview
Build a high-performance AI data pipeline and retrieval engine for **Souli**, an emotional wellbeing platform. The system processes YouTube coaching links from a CSV, extracts high-granularity diagnostic data using **Llama 3**, and stores it in a **Qdrant** vector database using a **Tiered Metadata Architecture**.

## 2. Technology Stack
- **Backend:** FastAPI (Python)
- **Frontend/UI:** Streamlit (for admin data management and sandbox testing)
- **AI Orchestration:** LangChain or LlamaIndex
- **LLM:** Llama 3 (via Ollama for local or Groq for cloud)
- **Vector Database:** Qdrant (Local Docker or Cloud)
- **Embeddings:** FastEmbed (optimized for Qdrant) or OpenAI `text-embedding-3-small`

## 3. Project Structure
```text
souli-engine/
├── app/
│   ├── main.py            # FastAPI entry point & routes
│   ├── services/
│   │   ├── extractor.py    # YT processing + Llama 3 Extraction logic
│   │   ├── qdrant_db.py    # Qdrant collection & metadata management
│   │   └── text_utils.py   # Transcript cleaning & normalization
│   └── models/
│       └── metadata.py     # Pydantic models for JSON schema
├── ui/
│   └── app.py             # Streamlit Dashboard
├── data/                  # Local storage for processed CSV exports
├── .env                   # Configuration & API Keys
├── .gitignore             # Git exclusion rules
├── requirements.txt       # Python dependencies
└── README.md              # Setup & execution guide

4. Core Feature SpecificationsA. Dynamic Data Ingestion (/process-csv)Input: Accept a CSV file with at least one column: yt_link.Transcript Processing: - Fetch transcripts using YoutubeLoader.Clean text: Remove timestamps, speaker labels, and marketing fluff.Multi-Row Extraction (Llama 3): - Analyze the transcript to identify 3 to 6 distinct use cases/problem statements.For each use case, generate a structured record following the Tiered Metadata Schema:Pillars: intervention_narrative (story), intervention_action (exercise), intervention_shift (one-liner).Atmosphere: tone and pacing of the coach.Overflow: Catch-all list for unique phrases or "gems" that don't fit pillars.Data Persistence: Save the flattened results into a new CSV for audit and bulk-upsert them into Qdrant.B. Qdrant Vector Storage StrategyCollection Setup: Create a Qdrant collection with a vector size corresponding to the chosen embedding model.Payload (Metadata): - Vectorized Field: Concatenate main_question and category for the embedding.Stored Payload: Store the entire Tiered JSON object (Pillars, Atmosphere, Overflow) in the Qdrant payload.Architecture Note: This avoids "null noise" as Qdrant handles sparse/dynamic payloads efficiently.C. Contextual Retrieval (/query)Search: Take a user's natural language input.Retrieve: Perform a similarity search in Qdrant ($k=3$).Response Logic: Return the matching Energy Nodes along with their associated metadata (Pillars/Atmosphere) so the frontend can "reconstruct" the coach's persona.
.env
# AI Provider (Choose Ollama or Groq)
LLM_TYPE=ollama 
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=your_groq_key_here

# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=souli_knowledge_base

# Settings
LOG_LEVEL=INFO

.gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.env

# Data & DB
data/*.csv
data/*.xlsx
qdrant_storage/

# OS
.DS_Store

6. UI Dashboard (Streamlit)
Tab 1: Ingestion. Upload CSV -> Progress Bar (Processing YT Links) -> Display Preview of extracted Rows -> Download link for the new CSV.

Tab 2: Chat Sandbox. Input a user struggle -> Display retrieved "Coaching Soul" metadata (The Story, The Action, and The Vibe).

7. Operational Instructions for the AI
Ensure the extractor.py uses a strict Few-Shot prompt to force Llama 3 to output the correct JSON structure for the metadata tiers.

In qdrant_db.py, use the qdrant-client library to handle upserts and similarity searches with high efficiency.

Provide a requirements.txt including: fastapi, uvicorn, qdrant-client, langchain, youtube-transcript-api, pandas, streamlit, and pydantic.