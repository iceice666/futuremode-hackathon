"""Router assembly + the error taxonomy from spec.md 2.6."""

from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..sidecar import SidecarFailed, SidecarTimeout, SidecarUnavailable

log = logging.getLogger(__name__)

STATUS_FOR_CODE = {
    "INVALID_PARAM": 400,
    "INVALID_CURSOR": 400,
    "FRAME_NOT_FOUND": 404,
    "NOT_FOUND": 404,
    "METHOD_NOT_ALLOWED": 405,
    "SIDECAR_UNAVAILABLE": 503,
    "SIDECAR_TIMEOUT": 504,
    "SIDECAR_FAILED": 502,
    "INTERNAL": 500,
}
"""spec.md 2.6. Every response pairs a code with exactly this status, so the
reverse lookup below can never emit a combination outside the table."""

CODE_FOR_STATUS = {
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
}
"""Starlette raises bare HTTPExceptions for routing failures, which carry no
code of ours. `FRAME_NOT_FOUND` must not be the answer for an unknown route:
that would tell the frontend a frame is missing when the path is simply wrong."""


class ApiError(Exception):
    """Raise this anywhere in a handler; the shape below is spec.md 2.6."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = STATUS_FOR_CODE.get(code, 500)


def error_response(code: str, message: str, status: int | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status if status is not None else STATUS_FOR_CODE.get(code, 500),
        content={"error": {"code": code, "message": message}},
        # spec.md 2.7 promises this header unconditionally, but an unhandled
        # exception is served by ServerErrorMiddleware, which sits *outside*
        # CORSMiddleware -- so a 500 would otherwise reach Bryan's dev server
        # as an opaque CORS failure instead of the message above. Setting it
        # here covers every error path; CORSMiddleware overwrites it with the
        # same value on the paths it does wrap.
        headers={"Access-Control-Allow-Origin": "*"},
    )


# Endpoint modules import ApiError from this package, so they are imported
# after it exists. Keep this block at the bottom.
from . import ask, events, frames, health, stream


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api")
    router.include_router(health.router)
    router.include_router(events.router)
    router.include_router(frames.router)
    router.include_router(ask.router)
    router.include_router(stream.router)
    return router


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # pydantic v2 body/query validation maps straight onto INVALID_PARAM.
        detail = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in detail.get("loc", ())[1:]) or "request"
        return error_response("INVALID_PARAM", f"{location}: {detail.get('msg', 'invalid')}")

    @app.exception_handler(SidecarUnavailable)
    async def _unavailable(_: Request, exc: SidecarUnavailable) -> JSONResponse:
        return error_response("SIDECAR_UNAVAILABLE", str(exc))

    @app.exception_handler(SidecarTimeout)
    async def _timeout(_: Request, exc: SidecarTimeout) -> JSONResponse:
        return error_response("SIDECAR_TIMEOUT", str(exc))

    @app.exception_handler(SidecarFailed)
    async def _failed(_: Request, exc: SidecarFailed) -> JSONResponse:
        return error_response("SIDECAR_FAILED", exc.message or exc.code)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Routing failures raised by Starlette itself. Anything we did not name
        # in CODE_FOR_STATUS is genuinely unexpected, so it keeps its own status
        # with INTERNAL rather than being dressed up as a taxonomy code.
        code = CODE_FOR_STATUS.get(exc.status_code, "INTERNAL")
        return error_response(code, str(exc.detail), status=exc.status_code)

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return error_response("INTERNAL", str(exc) or exc.__class__.__name__)
