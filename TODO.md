# TODO Items for Holocron

## High Priority

- [ ] **API Key Authentication**: Wire up `X-API-Key` header validation when `settings.api_key` is set
  - Add a FastAPI dependency in `src/app/deps.py` that checks the header
  - Apply to all endpoints except `/health`
  - Return 401 Unauthorized if key is missing or invalid

## Medium Priority

- [ ] **Bootstrap Schema Enhancement**: Update `scripts/bootstrap_schema.py` to cross-reference configured relationships in `relationships.yaml` against stored relationships, highlighting unused or missing labels

- [ ] **RSS Retry Logic**: Add exponential backoff retry for RSS feed fetching to handle transient network failures

- [ ] **Graph Endpoint Pagination**: Add `limit` and filtering parameters to `/graph` to prevent memory issues with large datasets

## Low Priority

- [ ] **INGEST_FEEDS Config Parser**: Fix list parsing from environment variables to support comma-separated values like `INGEST_FEEDS=fr/lemonde,es/elpais`

- [ ] **Metrics/Observability**: Add Prometheus metrics or OpenTelemetry instrumentation for production monitoring

- [ ] **Model Download Script**: Create a helper script to pre-download HuggingFace models for faster first-run experience

