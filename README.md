# Souli Data Ingestion & Retrieval System

A high-performance AI data pipeline for **Souli** — an emotional wellbeing platform.

The system extracts **3–6 structured coaching insights (EnergyNodes)** per YouTube video using **Llama 3**, stores them in **Qdrant**, and exposes a FastAPI backend + Streamlit admin dashboard.

---

## 🗂 Project Structure

```
souli-data-ingestion/
├── app/
│   ├── main.py               # FastAPI routes
│   ├── models/
│   │   └── metadata.py       # Pydantic: EnergyNode / Pillars / Atmosphere
│   └── services/
│       ├── extractor.py      # YouTube → Llama 3 → EnergyNodes
│       ├── qdrant_db.py      # Qdrant upsert & similarity search
│       └── text_utils.py     # Transcript fetch & clean
├── ui/
│   └── app.py                # Streamlit Admin Dashboard
├── data/                     # CSV exports (git-ignored)
├── tests/
│   └── test_extractor_smoke.py
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
| Ollama | Latest | Local Llama 3 |

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
# Use "ollama" for local or "groq" for cloud Llama 3
LLM_TYPE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Or for Groq cloud:
# LLM_TYPE=groq
# GROQ_API_KEY=your_key_here

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=souli_knowledge_base
```

### 4 — Start Qdrant (Docker)

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```
```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```



### 5 — Start Ollama + Pull Llama 3

```bash
ollama serve         # starts the Ollama server
ollama pull llama3   # downloads the model (~4GB)
```

### 6 — Start FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7 — Start Streamlit UI

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
  -d '{"query": "I feel exhausted and nothing is ever enough", "k": 3}'
```

---

## 🧠 Tiered Metadata Architecture

Each coaching video produces **3–6 EnergyNodes**, one per distinct coaching use-case:

```json
{
  "video_id": "abc123",
  "video_url": "https://youtube.com/watch?v=abc123",
  "main_question": "How do I stop feeling overwhelmed?",
  "category": "Anxiety",
  "pillars": {
    "intervention_narrative": "Anxiety is like a smoke alarm ...",
    "intervention_action": "3-minute morning body scan practice.",
    "intervention_shift": "Move from 'what is wrong with me' to 'what is my body trying to tell me.'"
  },
  "atmosphere": {
    "tone": "warm and reassuring",
    "pacing": "slow and deliberate"
  },
  "overflow": ["Your nervous system is not broken.", "Build a language between you and your body."]
}
```

**Vectorized:** `main_question + category`  
**Stored in Qdrant payload:** Full JSON above (sparse-safe — no null noise)

---

## 🧪 Smoke Test

```bash
# Requires Ollama running with llama3 pulled (or GROQ_API_KEY set)
python -m pytest tests/test_extractor_smoke.py -v
```

---

## 🌊 Using Groq Instead of Ollama

```env
LLM_TYPE=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama3-8b-8192
```

Get a free key at [console.groq.com](https://console.groq.com).
