"""
使用uvicorn，启动fastapi后台服务。
"""
import uvicorn
from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from app.api.errors.http_error import http_error_handler
from app.api.routes.recive_request import router as api_router
from app.api.config.config import ALLOWED_HOSTS, API_PREFIX, DEBUG, PROJECT_NAME, VERSION
from app.api.routes.events import create_start_app_handler, create_stop_app_handler

# app设置函数，返回app对象
def get_application() -> FastAPI:
    # app对象
    application = FastAPI(title=PROJECT_NAME, debug=DEBUG, version=VERSION)

    # 中间件设置
    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_HOSTS or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # app开启关闭执行事件设置
    application.add_event_handler("startup", create_start_app_handler())
    application.add_event_handler("shutdown", create_stop_app_handler())

    # http异常处理
    application.add_exception_handler(HTTPException, http_error_handler)

    # 添加路由，路由来自别的模块。
    application.include_router(api_router, prefix=API_PREFIX)

    return application


app = get_application()

if __name__ == "__main__":
    # 启动后台服务
    uvicorn.run(app, port=8001)
