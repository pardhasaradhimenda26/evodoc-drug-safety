"""
EvoDoc Clinical Drug Safety Engine — FastAPI Application
Production-grade API with full validation, logging, and monitoring.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cache import get_cache
from engine import DrugSafetyEngine, OllamaClient
from models import (
    DrugSafetyRequest,
    DrugSafetyResponse,
    HealthResponse,
    ErrorResponse,
)

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "biomistral")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
API_VERSION = "2.0.0"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evodoc.main")

# ─────────────────────────────────────────────
# APP STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

engine: DrugSafetyEngine = None
start_time_epoch: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, start_time_epoch
    start_time_epoch = time.time()

    logger.info("=" * 60)
    logger.info(f"  EvoDoc Drug Safety Engine v{API_VERSION} starting up")
    logger.info(f"  LLM Backend : {OLLAMA_URL} | Model: {MODEL_NAME}")
    logger.info(f"  Cache TTL   : {CACHE_TTL}s")
    logger.info("=" * 60)

    ollama_client = OllamaClient(base_url=OLLAMA_URL, model=MODEL_NAME)
    engine = DrugSafetyEngine(ollama_client=ollama_client)

    llm_ok = await ollama_client.is_available()
    if llm_ok:
        logger.info(f"✅ LLM '{MODEL_NAME}' is available at {OLLAMA_URL}")
    else:
        logger.warning(f"⚠️  LLM unavailable — fallback rule engine will be used")

    yield

    # Cleanup
    cache = get_cache()
    removed = cache.cleanup_expired()
    logger.info(f"Shutdown: cleared {removed} expired cache entries")


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="EvoDoc Clinical Drug Safety Engine",
    description=(
        "AI-powered drug interaction checker for Indian clinics. "
        "Accepts proposed medicines + patient history, returns structured safety assessment. "
        "Medical-grade accuracy with full audit trail."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MIDDLEWARE: Request ID + Timing
# ─────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = req_id
    start = time.time()

    response = await call_next(request)

    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
    logger.info(f"[{req_id}] {request.method} {request.url.path} → {response.status_code} ({elapsed_ms}ms)")
    return response


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{req_id}] Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please retry or contact EvoDoc support.",
            request_id=req_id,
        ).model_dump(),
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "EvoDoc Clinical Drug Safety Engine",
        "version": API_VERSION,
        "status": "operational",
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
async def health_check():
    cache = get_cache()
    llm_available = await engine.llm.is_available()
    return HealthResponse(
        status="healthy" if llm_available else "degraded",
        llm_available=llm_available,
        cache_backend="in-memory",
        cache_size=cache.size,
        model_name=MODEL_NAME,
        version=API_VERSION,
        uptime_seconds=round(time.time() - start_time_epoch, 1),
    )


@app.get(
    "/cache/stats",
    summary="Cache performance statistics",
    tags=["System"],
)
async def cache_stats():
    cache = get_cache()
    return cache.stats()


@app.post(
    "/cache/clear",
    summary="Clear all cache entries",
    tags=["System"],
)
async def clear_cache():
    cache = get_cache()
    count = cache.clear()
    return {"cleared": count, "message": f"Removed {count} cache entries"}


@app.post(
    "/analyze",
    response_model=DrugSafetyResponse,
    status_code=status.HTTP_200_OK,
    summary="Clinical Drug Safety Analysis",
    description=(
        "Core endpoint. Accepts proposed medicines and patient history. "
        "Returns structured safety assessment: drug interactions, allergy alerts, "
        "contraindications, risk score, and prescription safety verdict."
    ),
    tags=["Clinical"],
    responses={
        200: {"description": "Safety assessment completed"},
        422: {"description": "Validation error in request data"},
        500: {"description": "Engine error"},
    },
)
async def analyze_drug_safety(
    request: DrugSafetyRequest,
    http_request: Request,
) -> DrugSafetyResponse:
    """
    **Clinical Drug Safety Analysis**

    Checks:
    - Drug-drug interactions (proposed × proposed, proposed × current)
    - Allergy alerts including drug class cross-reactivity
    - Drug-condition contraindications
    - Overall patient risk score (0–100)
    - Prescription safety verdict

    **LLM**: BioMistral-7B (medical-specific). Falls back to rule-based engine if unavailable.

    **Caching**: Results are cached for 1 hour by deterministic drug hash.
    """
    # Inject request ID if not provided
    if not request.request_id:
        request.request_id = getattr(http_request.state, "request_id", None)

    try:
        response = await engine.analyze(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Engine error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Drug safety analysis failed. Using fallback — please retry.",
        )


@app.post(
    "/analyze/batch",
    summary="Batch analysis for multiple patients",
    tags=["Clinical"],
    description="Analyze drug safety for up to 10 patients in a single request.",
)
async def analyze_batch(
    requests: list[DrugSafetyRequest],
    http_request: Request,
):
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")

    results = []
    for req in requests:
        if not req.request_id:
            req.request_id = str(uuid.uuid4())[:8]
        try:
            result = await engine.analyze(req)
            results.append({"request_id": req.request_id, "result": result.model_dump()})
        except Exception as e:
            results.append({
                "request_id": req.request_id,
                "error": str(e),
                "result": None,
            })

    return {"batch_count": len(results), "results": results}


@app.get(
    "/interactions/fallback",
    summary="View fallback drug interaction dataset",
    tags=["Reference"],
)
async def get_fallback_dataset():
    """Returns the full fallback interaction dataset used when LLM is unavailable."""
    from engine import FALLBACK_DATA
    return {
        "total": len(FALLBACK_DATA["interactions"]),
        "metadata": FALLBACK_DATA["metadata"],
        "interactions": FALLBACK_DATA["interactions"],
    }


@app.get(
    "/drug-classes",
    summary="View known drug class mappings",
    tags=["Reference"],
)
async def get_drug_classes():
    """Returns the drug-to-class mapping used for allergy cross-reactivity detection."""
    from engine import DRUG_CLASS_MAP
    return {"classes": DRUG_CLASS_MAP}
