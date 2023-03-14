from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.db.elasticsearchwzh import es_new, es_old


async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    es_new.close()
    es_old.close()
    return JSONResponse({"errors": [exc.detail]}, status_code=exc.status_code)
