# Souli AI — Data Ingestion & Retrieval System

**Souli AI Mobile Application** — an AI-powered emotional wellness companion that supports users through daily emotional challenges using safe emotional expression, personalized insights, and short guided practices.

This pipeline extracts **3–6 structured coaching insights (EnergyNodes)** per YouTube video, stores them in **Qdrant**, and exposes a FastAPI backend for the Souli RAG system.

---

## 🗂 Project Structure

```
souli-data-ingestion/
├── app/
│   ├── main.py               # FastAPI routes (v2.0.0)
│   ├── models/
│   │   └── metadata.py       # Pydantic: EnergyNode / DiagnosticLayer / Pillars / Atmosphere
│   └── services/
│       ├── extractor.py      # YouTube → LLM → EnergyNodes
│       ├── qdrant_db.py      # Qdrant upsert & similarity search
│       └── text_utils.py     # Transcript fetch & clean
├── ui/
│   └── app.py                # Streamlit Admin Dashboard
├── data/                     # CSV exports (git-ignored)
├── tests/
│   └── test_extractor_smoke.py
├── Souli_EnergyFramework_PW.xlsx  # Source framework (2 sheets)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1 — Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend |
| Docker | Latest | Qdrant |
| Groq API Key | — | Cloud Llama 3 (recommended) |

### 2 — Clone & Install

```bash
git clone <your-repo-url>
cd souli-data-ingestion

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Use "groq" for cloud Llama 3 (recommended) or "ollama" for local
LLM_TYPE=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Or for local Ollama:
# LLM_TYPE=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=souli_knowledge_base
```

### 4 — Start Qdrant (Docker)

```bash
# Linux / macOS
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```
```powershell
# Windows PowerShell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 5 — Start FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6 — Start Streamlit UI

```bash
streamlit run ui/app.py
```

Open: [http://localhost:8501](http://localhost:8501)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/collection-info` | Qdrant collection stats |
| `POST` | `/process-csv` | Upload CSV → extract → upsert |
| `GET` | `/download-csv` | Download last extraction as CSV |
| `POST` | `/query` | Semantic similarity search |

### Example: Process CSV

```bash
curl -X POST http://localhost:8000/process-csv \
  -F "file=@my_links.csv"
```

CSV format:
```csv
yt_link
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I feel emotionally exhausted and nothing is ever enough", "k": 3}'
```

---

## 🧠 EnergyNode Schema (v2.0)

Each coaching video produces **3–6 EnergyNodes**. The schema has two logical tiers:

| Tier | Field | Purpose |
|------|-------|---------|
| **Routing** | `diagnostic_layer` | Decides *which* energy node the user is in |
| **Response** | `pillars`, `atmosphere`, `overflow` | Decides *how* to talk after routing |

```json
{
  "video_id": "abc123",
  "video_url": "https://youtube.com/watch?v=abc123",
  "main_question": "How do I stop feeling overwhelmed?",
  "category": "Anxiety",

  "diagnostic_layer": {
    "related_inner_issues": "chronic stress, body-mind disconnection, hypervigilant nervous system, suppressed fear",
    "reality_commitment_check": "Are you willing to slow down and listen to what your body is telling you?",
    "hidden_benefit": "staying in overdrive keeps a sense of productivity and avoids sitting with uncomfortable feelings",
    "energy_node": "hypervigilant_energy"
  },

  "pillars": {
    "intervention_narrative": "Anxiety is like a smoke alarm — it signals 'check the kitchen', not 'the house is on fire'.",
    "intervention_action": "3-minute morning body scan — notice sensations without trying to fix them.",
    "intervention_shift": "Move from 'what is wrong with me' to 'what is my body trying to tell me.'"
  },
  "atmosphere": {
    "tone": "warm and reassuring",
    "pacing": "slow and deliberate"
  },
  "overflow": ["Your nervous system is not broken.", "Build a language between you and your body."]
}
```

### Embedding Strategy

**Vectorized:** `main_question + category + related_inner_issues + reality_commitment_check + hidden_benefit + energy_node`

The full `diagnostic_layer` is included in the embedding so that semantic search lands on the correct energy block — not just the surface question. This is the **deciding factor** for RAG retrieval.

**Stored in Qdrant payload:** Full JSON above.

---

## 🔬 DiagnosticLayer Fields (from Souli_EnergyFramework_PW.xlsx)

| Field | Source Column | Description |
|-------|--------------|-------------|
| `related_inner_issues` | `Related Inner Issues` | Root psychological dynamics driving the struggle |
| `reality_commitment_check` | `Reality Commitment Check` | Yes/no question testing user readiness to change |
| `hidden_benefit` | `Hidden Psychological / Emotional Benefit` | Unconscious secondary gain from staying stuck |
| `energy_node` | `energy_node/energy block behind it` | Canonical snake_case energy-block label |

Known `energy_node` values: `blocked_energy`, `outofcontrol_energy`, `scattered_energy`, `depleted_energy`, `collapsed_energy`, `hypervigilant_energy`, `disconnected_energy`, `wounded_energy`.

---

## 🧪 Smoke Tests

```bash
# Requires GROQ_API_KEY or Ollama running with llama3
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Tests cover: list return, minimum node count, EnergyNode type validation, all core field non-empty checks, **diagnostic_layer field checks**, embed_text content checks, and video metadata preservation.

---

## 🌊 Using Groq (Recommended)

```env
LLM_TYPE=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free key at [console.groq.com](https://console.groq.com).
