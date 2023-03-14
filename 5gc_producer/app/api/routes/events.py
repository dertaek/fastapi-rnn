"""
app执行事件模块，分别有：
    1、起始和结束事件
    2、定期执行的预测以及异常检测事件
"""

from typing import Callable
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.pool import ProcessPoolExecutor
from app.get_data.post_predict_request import post_predict_request, post_abnormal_request
from app.get_data.post_predict_request import yaml_decode
import datetime
from app.db.elasticsearchwzh import es


# 起始事件
def create_start_app_handler() -> Callable:
    def start_app() -> None:
        logger.info("预启动，前置处理")

        job_info_dict = yaml_decode("app/vol", "job_config.yml")['SubmitPredictJobRequest']
        abnormal_info_dict = yaml_decode("app/vol", "job_config.yml")['SubmitAbnormalJobRequest']
        # job_info_dict = yaml_decode(os.path.abspath("."), "job_config.yml")['SubmitPredictJobRequest']
        # abnormal_info_dict = yaml_decode(os.path.abspath("."), "job_config.yml")['SubmitAbnormalJobRequest']
        logger.info("读取任务参数")

        scheduler = _get_scheduler(job_info_dict, abnormal_info_dict)
        logger.info("注册scheduler")

        scheduler.start()

    return start_app


# 结束事件
def create_stop_app_handler() -> Callable:
    @logger.catch
    async def stop_app() -> None:
        await es.close()
        logger.info("预关闭，后置处理")
    return stop_app


# 定期执行的预测以及异常检测事件
def _get_scheduler(job_info_dict, abnormal_info_dict):
    executors = {
        'processpool': ProcessPoolExecutor(3)
    }
    scheduler = AsyncIOScheduler(executors=executors)
    # 预测事件
    scheduler.add_job(post_predict_request, 'interval', hours=4.5, kwargs=job_info_dict, next_run_time=datetime.datetime.now())
    # 异常检测事件
    scheduler.add_job(post_abnormal_request, 'interval', minutes=5, kwargs=abnormal_info_dict, next_run_time=datetime.datetime.now() + datetime.timedelta(minutes=5))
    return scheduler
