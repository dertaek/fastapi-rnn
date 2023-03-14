from typing import Callable
from loguru import logger
from app.db.elasticsearchwzh import es_new, es_old


def create_start_app_handler() -> Callable:
    def start_app() -> None:
        logger.info("预启动，前置处理")
    return start_app


def create_stop_app_handler() -> Callable:
    @logger.catch
    async def stop_app() -> None:
        es_new.close()
        es_old.close()
        logger.info("预关闭，后置处理")

    return stop_app



