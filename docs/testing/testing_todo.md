# Holocron Testing To-Do (Day Plan)

Follow this checklist to validate every requirement end-to-end.

## Environment Prep
- [/] Confirm Python 3.11, uv, pytest installed
- [/] Start Ollama (`ollama serve`) and pull `llama3`
- [/] Copy `.env.example` → `.env`; set `LOG_LEVEL=DEBUG`
- [ ] Download HF models (first run of NER/translation or via cache script)

## Database & Storage
- [ ] Drop & init DB: `python -m src.app.storage.db --drop --init`
- [ ] Inspect tables (`articles`, `entities`, `relationships`) via SQLite browser or `sqlite3`
- [ ] Validate unique constraint: attempt duplicate entity norm per article ⇒ expect failure

## RSS Ingestion Script
- [ ] Run `python scripts/run_ingest_once.py --feed fr/lemonde --limit 3`
- [ ] Check logs for success counts and timings
- [ ] Verify DB contains 3 new articles with entities/relationships

## Scheduler Verification
- [ ] Set `.env`: `SCHEDULER_ENABLED=true`, `INGEST_INTERVAL_MINUTES=1`
- [ ] Start API (`uvicorn src.app.main:app --reload`)
- [ ] Observe at least two scheduler runs in logs (success/failure counts)
- [ ] Confirm no duplicate articles (unique URL enforcement)

## Deterministic NLP
- [ ] Run `pytest tests/test_nlp.py`
- [ ] Manually call NER on sample text (e.g., via REPL) ⇒ only allowed entity types
- [ ] Translate list of French names ⇒ verify English output

## Relationship Classifier
- [ ] With Ollama running: call classifier on sample PERSON↔ORG ⇒ expect label
- [ ] Stop Ollama, set `OPENAI_API_KEY`, rerun ⇒ OpenAI fallback works
- [ ] Remove API keys, rerun ⇒ heuristic fallback returns `None` or label
- [ ] Enable DEBUG logs, confirm reduced label checks per pair

## API Contract Tests
- [ ] `POST /entities` minimal payload ⇒ 201, proper response shape
- [ ] `POST /entities` full payload (title+text+url+lang) ⇒ 201, bilingual fields present
- [ ] `POST /entities` missing title ⇒ 422 validation error
- [ ] `GET /articles` ⇒ list of stored articles
- [ ] `GET /entities?type=PERSON` ⇒ filtered entities
- [ ] `GET /relationships?rel_type=met_with` ⇒ filtered relationships
- [ ] `GET /graph` ⇒ nodes/edges arrays; handles empty state
- [ ] `GET /health` ⇒ status ok

## Schema Governance
- [ ] Run `python scripts/bootstrap_schema.py --limit 10`
- [ ] Review distribution + sample contexts for human QA
- [ ] Add temporary label to `relationships.yaml`, rerun ingest ⇒ ensure schema reloads

## Logging & Error Handling
- [ ] Trigger RSS failure (disconnect network) ⇒ ERROR log with stack
- [ ] Submit payload with empty text ⇒ 400 response, descriptive message
- [ ] Toggle `LOG_LEVEL=INFO` ⇒ confirm verbosity changes

## Docker & Deployment
- [ ] `docker build -t holocron-news-intel .`
- [ ] `docker run -p 8080:8080 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 holocron-news-intel`
- [ ] Wait for health check ⇒ container status `healthy`
- [ ] Optional: `docker-compose up` ⇒ confirm Postgres + Ollama integration

## Edge Cases & Regression
- [ ] Process article with no entities ⇒ response contains empty lists, no errors
- [ ] Confirm duplicate relationships are deduped (no double edges)
- [ ] Set `lang=es` with Spanish headline ⇒ translation uses es→en model

## Documentation & Evidence
- [ ] Record curl/Postman requests covering all endpoints
- [ ] Save logs/screenshots for scheduler run, bootstrap report, Docker health
- [ ] Summarize pass/fail linked to initial project requirements


