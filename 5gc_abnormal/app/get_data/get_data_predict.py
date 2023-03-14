import pandas as pd
from pandas.errors import EmptyDataError

from app.db.es_redis_config import ES_KPI_SET
from loguru import logger
from app.model.predict import get_data_run_RNN_predict, get_rest_data_run_RNN_predict
import yaml
import os
import time


volPath = "app/vol/config"


# 预测任务主函数
def get_data_predict(*, resource_type, resid, input_idx_name, predict_method) -> None:
    logger.info("预测任务开始执行！！")
    if predict_method == "RNN":
        kpi = ES_KPI_SET[resource_type]
        config_dict = yaml_decode(volPath, "RNN.yml").get(resource_type)
        # config_dict = yaml_decode(os.path.abspath("."), "RNN.yml").get(resource_type)
        if os.path.exists(f"/app/vol/history_data/{resid}.csv") and fix_cache(resid):
            get_rest_data_run_RNN_predict(resid, idx_name=input_idx_name, type_=resource_type, kpi=kpi,
                                 config_dict=config_dict)
        else:
            get_data_run_RNN_predict(resid, idx_name=input_idx_name, type_=resource_type, kpi=kpi,
                                 config_dict=config_dict)


def yaml_decode(path, name) -> dict:
    yamlPath = os.path.join(path, name)
    with open(yamlPath, 'r', encoding='UTF-8') as rnn_config:
        config = rnn_config.read()
        config_dict = yaml.safe_load(config)
    return config_dict


def yaml_save(data: dict, path, name) -> None:
    yamlPath = os.path.join(path, name)
    with open(yamlPath, 'w', encoding='UTF-8') as rnn_config:
        yaml.dump(data, rnn_config)


def fix_cache(resid):
    logger.debug("开始验证缓存正确性")
    try:
        origin_data = pd.read_csv(f"/app/vol/history_data/{resid}.csv", index_col=0)
    except EmptyDataError:
        logger.debug("验证结果：缓存为空")
        return False
    
    col = origin_data.shape[0]
    if col == 0:
        logger.debug("验证结果：缓存为空")
        return False

    count = 0
    while True:
        try:
            time.strptime(origin_data.index[-1 - count], "%Y-%m-%d %H:%M:%S")
            break
        except ValueError:
            count += 1
            if count >= col:
                logger.debug("验证结果：缓存时间格式全部错误")
                return False

    if count > 0:
        origin_data = origin_data.iloc[:-count, :]
        origin_data.to_csv(f"/app/vol/history_data/{resid}.csv")
        logger.debug(f"验证结果：缓存时间格式错误，错误条数{count}。缓存已修正")

    return True