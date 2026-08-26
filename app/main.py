"""FastAPI application wiring all seven endpoints."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import Conflict, InvalidInput
from app.endpoints import (
    adapt,
    bqml,
    build_corpus,
    pipeline,
    promote,
    quantize,
    verify_bundle,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ga8")

app = FastAPI(title="Deterministic ML API", version="1.0.0")


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code}, media_type="application/json")


async def _json_body(request: Request):
    raw = await request.body()
    import json

    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
        )
    except Exception:
        raise InvalidInput()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/build-corpus")
async def build_corpus_route(request: Request):
    body = await _json_body(request)
    result = build_corpus.handle(body)
    return JSONResponse(content=result)


@app.post("/bqml")
async def bqml_route(request: Request):
    body = await _json_body(request)
    result = bqml.handle(body)
    return JSONResponse(content=result)


@app.post("/promote")
async def promote_route(request: Request):
    body = await _json_body(request)
    result = promote.handle(body)
    return JSONResponse(content=result)


@app.post("/adapt")
async def adapt_route(request: Request):
    body = await _json_body(request)
    result = adapt.handle(body)
    return JSONResponse(content=result)


@app.post("/quantize")
async def quantize_route(request: Request):
    body = await _json_body(request)
    result = quantize.handle(body)
    return JSONResponse(content=result)


@app.post("/pipeline")
async def pipeline_route(request: Request):
    body = await _json_body(request)
    result = pipeline.handle(body)
    return JSONResponse(content=result)


@app.post("/verify-bundle")
async def verify_bundle_route(request: Request):
    body = await _json_body(request)
    result = verify_bundle.handle(body)
    return JSONResponse(content=result)


@app.exception_handler(InvalidInput)
async def invalid_input_handler(request: Request, exc: InvalidInput):
    return _error(400, "INVALID_INPUT")


@app.exception_handler(Conflict)
async def conflict_handler(request: Request, exc: Conflict):
    return _error(409, exc.code)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, "INVALID_INPUT" if exc.status_code == 400 else "NOT_FOUND" if exc.status_code == 404 else "METHOD_NOT_ALLOWED" if exc.status_code == 405 else "ERROR")


@app.exception_handler(Exception)
async def unexpected_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")
    return _error(500, "INTERNAL_ERROR")
