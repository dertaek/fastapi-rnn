"""
向5gc_consumer发送请求模块
"""
from app.db.es_redis_config import ES_DEVICE_SETS, ES_KPI_SET
from loguru import logger
from app.db.elasticsearchwzh import get_es_indices, get_resid_list
import yaml
import os
import json
import requests
import time
from datetime import datetime

curPath = os.path.dirname(os.path.realpath(__file__))


# class AsyncRequests(object):
#     def __init__(self, url):
#         self.url = url
#
#     async def post(self, params):
#         return requests.post(self.url, json.dumps(params))
#
#     async def async_post(self, params):
#         response = await self.post(params)
#         return response
#
#     async def many_async_post(self, params_list):
#         tasks = [asyncio.create_task(self.async_post(params)) for params in params_list]
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(asyncio.wait(tasks))
#         return [task.result() for task in tasks]
#
#
# async def post_predict_request(*, resource_type, predict_method, predict_history) -> None:
#     logger.info("任务开始执行！！")
#     # 得到索引
#     input_idx_name = await get_es_indices(predict_history, time.strftime("%Y-%m-%d", time.localtime()))
#     # input_idx_name = await get_es_indices(predict_history, "2022-03-20")
#     if len(input_idx_name) != 0:
#         logger.info("es_indx:" + input_idx_name)
#         params_list = []
#         for type_ in resource_type:
#             dev_list = ES_DEVICE_SETS[type_]
#             kpi = ES_KPI_SET[type_]
#             logger.debug(f"网元类型：{type_} kpi指标：{kpi} 设备类型：{dev_list}")
#             res_id_list = await get_resid_list(input_idx_name, dev_list, type_)
#             for res_id in res_id_list:
#                 params = {
#                         "type_": type_,
#                         "res_id": res_id["key"],
#                         "input_idx_name": input_idx_name,
#                         "method": predict_method,
#
#                 }
#                 params_list.append(params)
#         logger.debug(params_list)
#         result = await AsyncRequests("http://ai5gc.ai-5gc:8018/api/predict").many_async_post(params_list)
#         print(result)

# 提交预测请求
def post_predict_request(*, resource_type, predict_method, predict_history, predict_sleep_time) -> None:
    logger.info("预测任务开始执行！！")
    # 得到索引
    input_idx_name = get_es_indices(datetime.now(), predict_history=predict_history)
    if input_idx_name:
        logger.info("es_indx:" + input_idx_name)
        for type_ in resource_type:
            dev_list = ES_DEVICE_SETS[type_]
            kpi = ES_KPI_SET[type_]
            logger.debug(f"网元类型：{type_} kpi指标：{kpi} 设备类型：{dev_list}")
            res_id_list = get_resid_list(input_idx_name, dev_list, type_)
            res_id_list.sort(key=lambda x: x["key"])
            logger.debug(f"res_id_list: {res_id_list}")
            for res_id in res_id_list:
                params = {
                        "type_": type_,
                        "res_id": res_id["key"],
                        "input_idx_name": input_idx_name,
                        "method": predict_method,
                }
                requests.post("http://ai5gc.ai-5gc:8018/api/predict", json.dumps(params))
                # requests.post("http://127.0.0.1:8000/api/predict", json.dumps(params))
                time.sleep(predict_sleep_time)
            time.sleep(5)


# 提交异常检测请求
def post_abnormal_request(*, resource_type, abnormal_sleep_time):
    logger.info("异常检测任务开始执行！！")
    # 得到索引
    input_idx_name = get_es_indices(datetime.now(), task_type="abnormal")
    # input_idx_name = await get_es_indices("2022-03-20", task_type="abnormal")
    if input_idx_name:
        logger.info("es_indx:" + input_idx_name)
        for type_ in resource_type:
            dev_list = ES_DEVICE_SETS[type_]
            kpi = ES_KPI_SET[type_]
            logger.debug(f"网元类型：{type_} kpi指标：{kpi} 设备类型：{dev_list}")
            params = {
                    "type_": type_,
                    "input_idx_name": input_idx_name,
            }
            requests.post("http://ai5gc-abnormal.ai-5gc:8002/api/abnormal", json.dumps(params))
            # requests.post("http://127.0.0.1:8000/api/abnormal", json.dumps(params))
            time.sleep(abnormal_sleep_time)


# yaml参数文件解析
def yaml_decode(path, name) -> dict:
    yamlPath = os.path.join(path, name)
    with open(yamlPath, 'r', encoding='UTF-8') as rnn_config:
        config = rnn_config.read()
        config_dict = yaml.safe_load(config)
    return config_dict


# yaml文件储存
def yaml_save(data: dict, path, name) -> None:
    yamlPath = os.path.join(path, name)
    with open(yamlPath, 'w', encoding='UTF-8') as rnn_config:
        yaml.dump(data, rnn_config)
