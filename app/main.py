from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import ROOT, settings
from app.core.logging import configure_logging, request_id_middleware
from app.db.base import Base, engine
from app.ml.model import model_service


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    if settings.model_autoload:
        model_service.load()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.middleware("http")(request_id_middleware)
app.include_router(router)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and int(length) > settings.max_lyric_chars * 4:
        return JSONResponse(status_code=413, content={"detail": "Request body is too large"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    errors = [{"type": item.get("type"), "loc": item.get("loc"), "msg": item.get("msg")} for item in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled request error")
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred"})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
