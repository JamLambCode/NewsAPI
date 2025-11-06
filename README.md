# Holocron News Intelligence API

Holocron ingests French news headlines, extracts entities through deterministic NLP, and uses tightly controlled LLM reasoning to classify relationships between entities. The API exposes an on-demand `/entities` endpoint alongside browsable entity, relationship, and article listings.

## Features

- Deterministic entity extraction via Hugging Face NuNER
- MarianMT translation for bilingual outputs
- Schema-governed relationship typing with an Ollama-first LLM client
- Optional OpenAI/OpenRouter fallbacks and heuristic safety nets
- Hybrid ingestion model with on-demand processing and scheduled RSS polling
- SQLite by default with an easy path to Postgres

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (or pip/poetry)
- [Ollama](https://ollama.com/download) with `llama3` model pulled

### Quick Start

1. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv venv && source .venv/bin/activate
   uv pip install -e '.[dev]'
   
   # Or using pip
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev]'
   ```

2. **Set up Ollama**
   ```bash
   # Install from https://ollama.com/download, then:
   ollama pull llama3
   ollama serve  # Run in background
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work for local dev)
   ```

4. **Initialize database**
   ```bash
   python -m src.app.storage.db --init
   ```

5. **Start the API**
   ```bash
   uvicorn src.app.main:app --reload --port 8080
   ```

6. **Test with sample data**
   ```bash
   # Ingest a few headlines from Le Monde
   python scripts/run_ingest_once.py --feed fr/lemonde --limit 3
   
   # Or test the on-demand endpoint
   curl -X POST http://localhost:8080/entities \
     -H "Content-Type: application/json" \
     -d '{"title":"Emmanuel Macron rencontre Olaf Scholz à Paris","feed":"fr/lemonde"}'
   
   # Browse the data
   curl http://localhost:8080/entities
   curl http://localhost:8080/relationships
   curl http://localhost:8080/graph
   ```

## Configuration

- `.env.example` documents the main knobs:
  - `DATABASE_URL`: defaults to SQLite, switch to Postgres when ready.
  - `SCHEDULER_ENABLED`, `INGEST_INTERVAL_MINUTES`, `INGEST_FEEDS`: control the background poller.
  - `OLLAMA_BASE_URL`, `OLLAMA_MODEL`: Ollama-first relationship classification.
  - `OPENAI_*`, `OPENROUTER_*`: optional fallbacks if remote APIs are available.
  - `API_KEY`: set to require `X-API-Key` headers (to be wired as needed).

## API Surface

- `POST /entities` — on-demand processing (PDF requirement). Accepts `title`, optional `text`, `feed`, and `url`; returns bilingual entities and typed relationships.
- `GET /articles` — latest stored articles.
- `GET /entities?type=PERSON` — browse recent entities filtered by type.
- `GET /relationships?rel_type=met_with` — retrieve recent relationships.
- `GET /graph` — condensed `{nodes, edges}` payload for quick visualization.
- `GET /health` — lightweight readiness probe.

Responses echo the original payload, include normalized/translated entity strings, and surface relationship endpoints with entity IDs to ease graph construction.

## Scheduler & Ingestion

- The APScheduler-based poller runs when `SCHEDULER_ENABLED=true`.
- Each configured feed slug (default `fr/lemonde`) is normalized via `src/app/ingest/rss.py`.
- Poller reuses the same deterministic pipeline as `/entities`, so on-demand and scheduled runs stay consistent.
- `scripts/run_ingest_once.py` gives you a manual “one-shot” ingest for demos or seeding data.

## Schema Governance

- Relationship definitions live in `relationships.yaml` and are loaded by the classifier.
- `scripts/bootstrap_schema.py` prints relationship distribution and sample contexts to guide schema evolution.
- Update the YAML file, commit the change, and redeploy to introduce new labels.

## Ollama-first LLM Strategy

- Default classifier uses local Ollama (`llama3`), making the project runnable without new API spend.
- OpenAI/OpenRouter fallbacks activate automatically when keys and model names are present.
- A final heuristic layer guards against total LLM failure modes.

## Project Structure

```text
holocron-news-intel/
├── pyproject.toml
├── .env.example
├── README.md
├── relationships.yaml
├── docker-compose.yml
├── scripts/
│   ├── bootstrap_schema.py
│   └── run_ingest_once.py
└── src/app/
    ├── __init__.py
    ├── config.py
    ├── deps.py
    ├── main.py
    ├── schemas.py
    ├── storage/
    │   ├── __init__.py
    │   ├── db.py
    │   ├── models.py
    │   └── crud.py
    ├── ingest/
    │   └── rss.py
    ├── nlp/
    │   ├── __init__.py
    │   ├── ner.py
    │   └── translate.py
    └── llm/
        └── relationships.py

## Usage Examples

```bash
# 1) Seed a few headlines (stores bilingual entities + relationships)
python scripts/run_ingest_once.py --feed fr/lemonde --limit 3

# 2) Call the spec-compliant endpoint with ad-hoc text
http POST :8080/entities title="La rencontre de Macron et Scholz" feed="fr/lemonde"

# 3) Browse derived knowledge
http :8080/entities type==PERSON
http :8080/relationships rel_type==met_with
http :8080/graph

# 4) Review schema health
python scripts/bootstrap_schema.py --limit 5
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Note: Some tests may require the full NER/translation models to be downloaded. For pure smoke tests without models:

```bash
pytest tests/test_nlp.py tests/test_relationships.py -v
```

## Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t holocron-news-intel .

# Run with SQLite
docker run -p 8080:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  holocron-news-intel

# Or use docker-compose for full stack (Postgres + Ollama)
docker-compose up
```

## Roadmap & TODOs

See `TODO.md` for the full list. Key items:

- **Wire API key authentication** (X-API-Key header validation)
- Expand relationship schema via bootstrap analysis
- Add retry logic for RSS fetching
- Implement graph endpoint pagination


