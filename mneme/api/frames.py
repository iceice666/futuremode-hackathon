"""GET /api/frames/{id}/thumb — spec.md 2.3.

Frame content is immutable once written, so we can cache hard.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from . import ApiError

router = APIRouter()

CACHE_CONTROL = "public, max-age=31536000, immutable"


# FastAPI's APIRoute does not imply HEAD from GET the way Starlette's Route
# does, and `curl -sI` is part of the acceptance list (backend.md 8.8).
@router.api_route("/frames/{frame_id}/thumb", methods=["GET", "HEAD"])
async def frame_thumb(
    request: Request, frame_id: str, full: str | None = Query(default=None)
) -> FileResponse:
    runtime = request.app.state.runtime
    row = await runtime.db.get_frame(frame_id)
    if row is None:
        raise ApiError("FRAME_NOT_FOUND", f"no frame {frame_id}")
    want_full = full not in (None, "", "0", "false", "False")
    relative = row.path if want_full else row.thumb_path
    # DB holds paths relative to --data-dir (spec.md 0); resolve here.
    path = runtime.config.data_dir / relative
    if not path.is_file():
        raise ApiError("FRAME_NOT_FOUND", f"frame {frame_id} has no file at {relative}")
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": CACHE_CONTROL}
    )
