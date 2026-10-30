import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from docattentionlens_api.config import Settings
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.receipts import FileReceiptStore
from docattentionlens_api.routers import jobs, methods, runtime
from docattentionlens_api.schemas.contracts import HealthStatus
from docattentionlens_api.repositories.documents import DocumentRepository
from docattentionlens_api.routers.documents import router as documents_router
from docattentionlens_api.routers.retrieval import router as retrieval_router
from docattentionlens_api.routers.generation import router as generation_router
from docattentionlens_api.routers.experiments import router as experiments_router
from docattentionlens_api.routers.evaluation import router as evaluation_router
from docattentionlens_api.routers.setup import router as setup_router
from docattentionlens_api.routers.marking import router as marking_router
from docattentionlens_api.routers.document_studies import router as document_studies_router
from docattentionlens_api.routers.attention_interventions import router as attention_interventions_router

logger = logging.getLogger("uvicorn.error")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Migrations are deliberately a separate, one-off operation.
        yield

    app = FastAPI(title="DocAttentionLens", version="0.1.0", lifespan=lifespan)
    app.state.repository = JobRepository(settings)
    app.state.settings = settings
    app.state.documents = DocumentRepository(settings)
    app.state.receipt_store = FileReceiptStore(settings.artifact_root)
    app.include_router(documents_router)
    app.include_router(retrieval_router)
    app.include_router(generation_router)
    app.include_router(experiments_router)
    app.include_router(evaluation_router)
    app.include_router(setup_router)
    app.include_router(marking_router)
    app.include_router(document_studies_router)
    app.include_router(attention_interventions_router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = uuid4().hex
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("request=%s method=%s status=%s duration_ms=%.1f", request_id, request.method, response.status_code, (perf_counter() - started) * 1000)
        return response

    @app.exception_handler(HTTPException)
    async def transport_error(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "Request does not match the supported contract.", "fields": [".".join(str(part) for part in e["loc"]) for e in exc.errors()]}})

    @app.exception_handler(psycopg.Error)
    async def unavailable_metadata(request: Request, exc: psycopg.Error):
        return JSONResponse(status_code=503, content={"error": {"code": "metadata_unavailable", "message": "Metadata is unavailable or migrations are required."}})

    @app.get("/health/live", response_model=HealthStatus, tags=["health"])
    def live():
        return HealthStatus(status="live")

    @app.get("/health/ready", response_model=HealthStatus, tags=["health"])
    def ready():
        dependencies = {}
        for name, probe in [("metadata", app.state.repository.ready), ("artifact_store", app.state.receipt_store.probe)]:
            try:
                probe()
                dependencies[name] = "ready"
            except (psycopg.Error, OSError, RuntimeError):
                dependencies[name] = "unavailable"
        status = "ready" if all(value == "ready" for value in dependencies.values()) else "unavailable"
        response = HealthStatus(status=status, dependencies=dependencies)
        return JSONResponse(status_code=200 if status == "ready" else 503, content=response.model_dump())

    for router in (methods.router, jobs.router, runtime.router):
        app.include_router(router)
    return app


app = create_app()
