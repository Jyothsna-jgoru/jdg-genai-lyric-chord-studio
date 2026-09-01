from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from fastapi import Request

from app.core.config import settings


request_id: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s request_id=%(request_id)s %(name)s %(message)s"))
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)


async def request_id_middleware(request: Request, call_next):
    value = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id.set(value)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = value
        return response
    finally:
        request_id.reset(token)

