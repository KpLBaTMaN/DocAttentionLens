# Development and verification

Run all commands below from the repository root.

## Overview

A React and FastAPI application for inspecting document retrieval, generator attention,
and controlled document experiments. The Python library supplies dataset and model
adapters, 67 analysis methods, capture formats, and numerical validation. PostgreSQL
stores metadata; immutable captures, uploads, investigations, and results use a separate
artifact volume. Existing evidence can be inspected without a GPU.

## Build and maintain

Use Python 3.12 and Node 22.15 or newer. On Windows:

    ./scripts/setup.ps1 -Python C:/Path/To/Python312/python.exe
    docker compose -f infra/compose.yaml up -d metadata
    ./.venv/Scripts/python.exe -m docattentionlens_api.migrate
    ./.venv/Scripts/python.exe -m uvicorn docattentionlens_api.main:app --host 127.0.0.1 --port 8000

Run the worker and frontend in additional terminals:

    ./.venv/Scripts/python.exe -m docattentionlens_worker.main
    npm run dev --prefix frontend

On Linux or macOS:

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements-dev.lock
    .venv/bin/python -m pip install --no-build-isolation --no-deps -e library -e backend
    npm ci --prefix frontend

Stop any Docker services that occupy native development ports first. Native defaults
use PostgreSQL on localhost:5432, data/, and var/artifacts. Generated local output
belongs in ignored var/. Production Docker builds accept only application sources
and locked build inputs; tests, local environments, datasets, and credentials stay out.

Repository responsibilities:

- backend/src: API, durable CPU/GPU workers, repositories, and SQL migrations.
- library/src: dataset/model adapters, capture contracts, methods, and evaluation.
- frontend/src: application UI and generated API types.
- infra: Compose configuration and environment example.
- tests, backend/tests, library/tests, frontend/unit, frontend/tests: regression tests
  and small known-answer fixtures.
- scripts: environment setup, contract exports, smoke/recovery checks, model conformance,
  and the example dataset builder.
- data: local benchmark exports (ignored) and the two committed examples.
- contracts/openapi.json: generated API contract.
- docs: design archive, implementation history, release verification, and detailed
  frontend behaviour notes (see docs/README.md). Not part of any build.

Each method owns its calculation and descriptor in the library's methods directory.
The registry publishes implementation references and checked primary sources. Some
method modules retain historical reference identifiers because saved intervention
plans verify their exact source hashes; these identifiers are not runtime file inputs.
Formula changes must continue to invalidate analysis caches and prepared plans.

## Verify

With the locked development environment installed:

    ./.venv/Scripts/python.exe -m pytest -q
    npm run api:generate --prefix frontend
    npm run build --prefix frontend
    npm run test:unit --prefix frontend
    npm run typecheck:tests --prefix frontend
    npm run test:render --prefix frontend

Set DOCATTENTIONLENS_TEST_DATABASE_URL to a disposable PostgreSQL database to include integration
tests. They create and drop isolated docattentionlens_test_* schemas. Without it, database tests
explicitly skip. Dataset integration tests also need the local benchmark exports, except
those for the committed examples (docs/examples.md).

The rendering suite starts a fixture-backed Vite server and needs no database or GPU.
It uses Edge on Windows; on Linux install Playwright Chromium first. On Windows,
ensure the Windows System32 directory is on PATH so Playwright can stop its fixture
server after testing. Visual snapshots
are platform-specific. The live browser suite uses the running application:

    npm run test:e2e --prefix frontend
    ./.venv/Scripts/python.exe scripts/smoke.py --restart
    ./.venv/Scripts/python.exe scripts/document_smoke.py --restart

Smoke and recovery checks create records and may restart services; use a disposable
deployment for release verification. Live capture tests skip when suitable evidence is
absent; DOCATTENTIONLENS_REQUIRE_SAVED_CAPTURES=1 makes missing evidence a failure.

Keep the accepted model-matrix baseline in var/verification/model-matrix-baseline.json
with operational backups; it is retained validation state, rather than a disposable log.
scripts/model_matrix.py checks registered models, pinned cache contents, kernels, and
real capture conformance. Use --skip-runs for checks without new GPU captures.
scripts/frontend_proxy_smoke.py verifies nginx recovery across API address changes
using isolated containers. Other conformance and recovery scripts cover retrieval,
generation, document methods, interventions, experiments, and evaluation.

After API changes, regenerate the OpenAPI and TypeScript contracts with api:generate.
After method changes, run python scripts/export_methods.py. Regenerate dataset fixtures
with python scripts/export_dataset_fixtures.py. Recompile Python locks with pip-tools
from backend/requirements.in and requirements-dev.in; retain the separate GPU lock.
Runtime dependencies and build/test tooling are pinned in their manifests and locks.
