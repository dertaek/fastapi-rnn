"""
http异常处理文件
"""
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.db.elasticsearchwzh import es


async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    await es.close()
    return JSONResponse({"errors": [exc.detail]}, status_code=exc.status_code)
