import time
from typing import List

import elasticsearch
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from app.db.es_redis_config import ES_SCROLL_TIMER, ES_SCROLL_ROW_NUM, ES_KPI_SET, ES_DEVICE_SETS, BEST
from loguru import logger
from app.db.es_redis_config import ES_RESOURCE_TYPES, ES_INDEX_CFG
import json
from datetime import datetime, timedelta
import numpy as np

es_old = None
es_new = None


def _init_es_conn_pool(tag):
    global es_old, es_new
    if es_old is None or es_old.ping() is False:
        es_hosts_old = ["http://****************", "http://*******************", "http://******************"]
        es_http_auth_old = ("************", "***************")
        es_old = Elasticsearch(es_hosts_old, http_auth=es_http_auth_old, timeout=60, max_retries=10, retry_on_timeout=True)
        logger.debug(f"建立ES_old连接: {es_hosts_old}")
    if es_new is None or es_new.ping() is False:
        es_hosts_new = ["http://****************", "http://*******************", "http://*****************0"]
        es_http_auth_new = ("**********", "******************")
        es_new = Elasticsearch(es_hosts_new, http_auth=es_http_auth_new, timeout=60, max_retries=10, retry_on_timeout=True)
        logger.debug(f"建立ES_new连接: {es_hosts_new}")
    if tag == "old":
        return es_old
    elif tag == "new":
        return es_new


def search_with_scroll(tag, **kwargs) -> list:
    """
    分页查询操作封装
    """
    es = _init_es_conn_pool(tag)
    res = es.search(**kwargs, scroll=ES_SCROLL_TIMER, size=ES_SCROLL_ROW_NUM, ignore=404, timeout="60s")

    if res.get("error") is not None:
        logger.debug(res)
        logger.error(f"查询ES发生错误，错误码{res['status']}")
        return []

    total = res["hits"]["total"]["value"]
    logger.info(f"匹配到的数据量为: {total} 条")
    if total == 0:
        return []
    results = res["hits"]["hits"]

    scroll_id = res["_scroll_id"]
    page_num = int((total - ES_SCROLL_ROW_NUM) / ES_SCROLL_ROW_NUM) + 1
    logger.info(f"分页查询，总页数: {page_num}")

    for i in range(page_num):
        try:
            res_scroll = es.scroll(scroll_id=scroll_id, scroll=ES_SCROLL_TIMER, request_timeout=60)
        except elasticsearch.exceptions.NotFoundError:
            es.clear_scroll(scroll_id=scroll_id, request_timeout=60)
            raise elasticsearch.exceptions.NotFoundError
        # es.clear_scroll(scroll_id=scroll_id, request_timeout=60)
        logger.debug(f"分页查询，当前页数: {i + 1} 已加载")
        # results是列表嵌套字典
        results = results + res_scroll["hits"]["hits"]

    # 清除es.scroll上下文
    es.clear_scroll(scroll_id=scroll_id, request_timeout=60)

    return results


# 创建索引
def create_index(index_name, mapping):
    es = _init_es_conn_pool(tag="new")
    if es.indices.exists(index=index_name, request_timeout=60):
        pass
    else:
        es.indices.create(index=index_name, body=mapping, timeout="60s")
        logger.info("创建索引: " + index_name)


# 批量写入
def bulk_to_es(actions):
    es = _init_es_conn_pool(tag="new")
    ret = bulk(es, actions)
    return ret


# 异常检测模块使用，获取某个五分钟的真实数据并按规则过滤
def read_input_by_restype(resource_type, input_idx_name, min_id):
    query = {
        "track_total_hits": "true",
        "query": {
            "bool": {
                "filter": {
                    "range": {
                        "RECORD_TIME": {
                            "gte": min_id,
                            "lte": min_id
                        }
                    }
                },
                "must": [
                    {
                        "terms": {
                            "DEVICE_SET": ES_DEVICE_SETS[resource_type]
                        }
                    },
                    {
                        "terms": {
                            "ITEM_CODE": ES_KPI_SET[resource_type]
                        }
                    }
                ]
            }
        }
    }
    if resource_type == "INTERFACE":
        query["query"]["bool"]["must_not"] = [{
            "match": {
                "ITEM_PARA": "-1"
            }
        }]
    else:
        query["query"]["bool"]["must"].append({
            "match": {
                "ITEM_PARA": "-1"
            }
        })
    params = {
        "_source": ",".join(['RESID', 'NODECODE', 'DEVICE_SET', 'ITEM_PARA', 'RECORD_TIME', 'ITEM_CODE', 'VALUE']),
        "size": ES_SCROLL_ROW_NUM
    }
    filter_path = ["hits.total.value", "hits.hits._source", "_scroll_id"]
    kwargs = {"index": input_idx_name, "body": query, "params": params, "filter_path": filter_path}
    truth_data = search_with_scroll(tag="old", **kwargs)

    if truth_data:
        source_data = [p["_source"] for p in truth_data]
        nf_df = pd.DataFrame(source_data)
        nf_df = nf_df.assign(
            DEVICE_TYPE=resource_type.upper(),
            MIN_ID=min_id)
        nf_df["VENDOR"] = nf_df["DEVICE_SET"].apply(lambda x: x.split("_")[3])
        nf_df = nf_df[['RESID', 'NODECODE', 'DEVICE_SET', 'ITEM_PARA', 'RECORD_TIME', 'ITEM_CODE', 'VALUE', "DEVICE_TYPE", "MIN_ID", "VENDOR"]]
        nf_df.drop_duplicates(keep="first", inplace=True)
    else:
        logger.debug("未取到数据")
        nf_df = pd.DataFrame(columns=['RESID', 'NODECODE', 'DEVICE_SET', 'ITEM_PARA', 'RECORD_TIME', 'ITEM_CODE', 'VALUE', "DEVICE_TYPE", "MIN_ID", "VENDOR"])

    nf_df = nf_df.loc[(nf_df['VALUE'].astype(float) >= 0) & (nf_df['VALUE'].astype(float) != 9223372036854775807)]
    nf_df = nf_df.loc[((nf_df['ITEM_CODE'].isin(BEST["MAX_BEST"] + BEST["MIN_BEST"])) & (nf_df['VALUE'].astype(float) <= 100)) | ~(nf_df['ITEM_CODE'].isin(BEST["MAX_BEST"] + BEST["MIN_BEST"]))]
    logger.info("数据加载完成")
    return nf_df


# 异常检测模块使用，按网元类型获取预测结果,并对异常数据重新计算上下限
def read_predict_result_by_restype(min_id):
    query = {
        "track_total_hits": "true",
        "query": {
            "bool": {
                "filter": {
                    "range": {
                        "recordTime": {
                            "gte": min_id,
                            "lte": min_id
                        }
                    }
                }
            }
        }
    }
    input_idx_name = ES_INDEX_CFG["output_index_5gc"]["future6h"].replace("{YYYYMMDD}", time.strftime("%Y%m%d", time.strptime(min_id, "%Y-%m-%d %H:%M:%S")))
    kwargs = {"index": input_idx_name, "body": query}
    predict_data = search_with_scroll(tag="new", **kwargs)
    if predict_data:
        source_data = [p["_source"] for p in predict_data]
        predict_df = pd.DataFrame(source_data)
        predict_df = predict_df[["resId", "itemPara", "recordTime", "createTime", "kpiId", "forecastValue", "kpiStd", "nodeCode", "deviceSet"]]
        predict_df.drop_duplicates(subset=["resId", "itemPara", "recordTime", "kpiId"], keep="first", inplace=True)
        pd.set_option("display.float_format", lambda x: '%.2f' % x)
        predict_df["forecastValue"] = predict_df["forecastValue"].astype(float)
        result = pd.DataFrame()
        for kpi in predict_df.drop_duplicates(subset=["kpiId"])["kpiId"]:
            if kpi in BEST["MAX_BEST"]:
                res = predict_df.loc[predict_df["kpiId"] == kpi, :]
                if res.loc[(predict_df["forecastValue"] > 70) & (predict_df["forecastValue"] <= 100), :].shape[0] > 0:
                    median = res.loc[(predict_df["forecastValue"] > 70) & (predict_df["forecastValue"] <= 100), :][
                        "forecastValue"].median()

                else:
                    median = 90
                res = res.assign(interval=abs(res["forecastValue"] - median) / (median + 0.001))
                normal = res.loc[((res["forecastValue"] < median) & (res["interval"] <= 0.25)) | (
                            res["forecastValue"] >= median), :]
                abnormal = res.loc[(res["forecastValue"] < median) & (res["interval"] > 0.25), :]
                normal = normal.assign(forecastValueLower=0.9 * normal["forecastValue"],
                                       forecastValueUpper=100)
                abnormal = abnormal.assign(
                    forecastValueLower=normal["forecastValueLower"].min() if normal.shape[0] != 0 else 70,
                    forecastValueUpper=100)
                result = pd.concat([result, abnormal, normal])

            elif kpi in BEST["SCALE_DECR"]:
                res = predict_df.loc[predict_df["kpiId"] == kpi, :]
                if predict_df.loc[(predict_df["kpiId"] == kpi) & (predict_df["forecastValue"] > 10000), :].shape[0] > 0:
                    median = res.loc[(predict_df["forecastValue"] > 10000), :]["forecastValue"].median()
                else:
                    median = 50000
                res = res.assign(interval=abs(res["forecastValue"] - median) / (median + 0.0001))
                abnormal = res.loc[(res["forecastValue"] < median) & (res["interval"] > 0.97), :]
                normal = res.loc[((res["forecastValue"] < median) & (res["interval"] <= 0.97)) | (
                            res["forecastValue"] >= median), :]
                abnormal = abnormal.assign(
                    forecastValueLower=normal["forecastValue"].min() if normal.shape[0] != 0 else 10000,
                    forecastValueUpper=normal["forecastValue"].max() if normal.shape[0] != 0 else 1000000)
                normal = normal.assign(forecastValueLower=0.7 * normal["forecastValue"],
                                       forecastValueUpper=10 * normal["forecastValue"])
                result = pd.concat([result, abnormal, normal])

            elif kpi == "5gcUpfSysAvgLoadCpu":
                res = predict_df.loc[predict_df["kpiId"] == kpi, :]
                if predict_df.loc[(predict_df["kpiId"] == kpi) & (predict_df["forecastValue"] < 80), :].shape[0] > 0:
                    median = res.loc[(predict_df["forecastValue"] < 80), :]["forecastValue"].median()
                else:
                    median = 10
                res = res.assign(interval=abs(res["forecastValue"] - median) / (median + 0.001))
                normal = res.loc[((res["forecastValue"] > median) & (res["interval"] <= 3.3)) | (
                            res["forecastValue"] <= median), :]
                abnormal = res.loc[(res["forecastValue"] > median) & (res["interval"] > 3.3), :]
                abnormal = abnormal.assign(forecastValueLower=0,
                                           forecastValueUpper=normal["forecastValue"].max() if normal.shape[
                                                                                                   0] != 0 else 80)
                normal = normal.assign(forecastValueLower=0, forecastValueUpper=np.where(
                    (1.5 * normal["forecastValue"] <= 95) & (1.5 * normal["forecastValue"] > 20),
                    1.5 * normal["forecastValue"], 95))
                result = pd.concat([result, abnormal, normal])

            elif kpi == "5gcUpfSysAvgLoadMe":
                res = predict_df.loc[predict_df["kpiId"] == kpi, :]
                if predict_df.loc[(predict_df["kpiId"] == kpi) & (predict_df["forecastValue"] < 80), :].shape[0] > 0:
                    median = res.loc[(predict_df["forecastValue"] < 80), ["forecastValue"]]["forecastValue"].median()
                else:
                    median = 20
                res = res.assign(interval=abs(res["forecastValue"] - median) / (median + 0.001))
                normal = res.loc[((res["forecastValue"] > median) & (res["interval"] <= 1.4)) | (
                            res["forecastValue"] <= median), :]
                abnormal = res.loc[(res["forecastValue"] > median) & (res["interval"] > 1.4), :]
                abnormal = abnormal.assign(forecastValueLower=0,
                                           forecastValueUpper=normal["forecastValue"].max() if normal.shape[
                                                                                                   0] != 0 else 80)
                normal = normal.assign(forecastValueLower=0, forecastValueUpper=np.where(
                    (1.5 * normal["forecastValue"] <= 95) & (1.5 * normal["forecastValue"] > 20),
                    1.5 * normal["forecastValue"], 95))
                result = pd.concat([result, abnormal, normal])

            elif kpi == "5gcSmfAvgSystemLoad" or kpi == "5gcAmfAvgSysLoad" or kpi == "Dcswitchcpuusage":
                res = predict_df.loc[predict_df["kpiId"] == kpi, :]
                if predict_df.loc[(predict_df["kpiId"] == kpi) & (predict_df["forecastValue"] < 80), :].shape[0] > 0:
                    median = res.loc[(predict_df["forecastValue"] < 80), ["forecastValue"]]["forecastValue"].median()
                else:
                    median = 20
                res = res.assign(interval=abs(res["forecastValue"] - median) / (median + 0.0001))
                normal = res.loc[
                         ((res["forecastValue"] > median) & (res["interval"] <= 2)) | (res["forecastValue"] <= median),
                         :]
                abnormal = res.loc[(res["forecastValue"] > median) & (res["interval"] > 2), :]
                abnormal = abnormal.assign(forecastValueLower=0,
                                           forecastValueUpper=normal["forecastValue"].max() if normal.shape[
                                                                                                   0] != 0 else 90)
                normal = normal.assign(forecastValueLower=0, forecastValueUpper=np.where(
                    (1.5 * normal["forecastValue"] <= 95) & (1.5 * normal["forecastValue"] > 20),
                    1.5 * normal["forecastValue"], 95))
                result = pd.concat([result, abnormal, normal])

            elif kpi == "5gcUpfPduSessModiFailNum":
                res = predict_df.loc[predict_df["kpiId"] == kpi, :]
                if predict_df.loc[(predict_df["kpiId"] == kpi) & (predict_df["forecastValue"] < 50), :].shape[0] > 0:
                    median = res.loc[(predict_df["forecastValue"] < 50), ["forecastValue"]]["forecastValue"].median()
                else:
                    median = 10
                res = res.assign(interval=abs(res["forecastValue"] - median) / (median + 0.001))
                normal = res.loc[((res["forecastValue"] > median) & (res["interval"] <= 5e5)) | (
                            res["forecastValue"] <= median), :]
                abnormal = res.loc[(res["forecastValue"] > median) & (res["interval"] > 5e5), :]
                abnormal = abnormal.assign(forecastValueLower=0,
                                           forecastValueUpper=normal["forecastValue"].max() if normal.shape[
                                                                                                   0] != 0 else 50)
                normal = normal.assign(forecastValueLower=0, forecastValueUpper=np.where(
                    (1.5 * normal["forecastValue"] <= 95) & (1.5 * normal["forecastValue"] > 20),
                    1.5 * normal["forecastValue"], 95))
                result = pd.concat([result, abnormal, normal])

        result = result.round({"forecastValue": 2, "forecastValueLower": 2, "forecastValueUpper": 2})
        result = result[["resId", "itemPara", "recordTime", "createTime", "kpiId", "forecastValue", "forecastValueUpper", "forecastValueLower", "nodeCode", "deviceSet"]]
        result = result.loc[(result["forecastValue"].astype(float) >= 0)]
        result = result.loc[((result['kpiId'].isin(BEST["MAX_BEST"] + BEST["MIN_BEST"])) & (
                    result['forecastValue'].astype(float) <= 100)) | ~(
            result['kpiId'].isin(BEST["MAX_BEST"] + BEST["MIN_BEST"]))]
        logger.debug(result)
        return result
    return pd.DataFrame(columns=["resId", "itemPara", "recordTime", "createTime", "kpiId", "forecastValue", "forecastValueUpper", "forecastValueLower", "nodeCode", "deviceSet"])


# 得到前五分钟，前十分钟，前十五分钟三个时间点
def get_min_id() -> List[str]:
    # es = _init_es_conn_pool(tag="old")
    # query = {
    #     "size": 1,
    #     "query": {
    #         "match_all": {}
    #     },
    #     "sort": {
    #         "RECORD_TIME": "desc"
    #     }
    # }
    # date_time = es.search(index=input_idx_name, body=query, ignore=404, timeout="60s")["hits"]["hits"][0]["_source"]["RECORD_TIME"]
    date_time_before5min = datetime.now() - timedelta(minutes=(int(datetime.strftime(datetime.now(), "%M")) % 5) + 25, seconds=int(datetime.strftime(datetime.now(), "%S")))
    date_time_before10min = date_time_before5min - timedelta(minutes=5)
    date_time_before15min = date_time_before10min - timedelta(minutes=5)
    # date_time_before5min = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=5)
    # date_time_before10min = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=10)
    # date_time_before15min = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=15)
    min_id, min_id_before5min, min_id_before10min = datetime.strftime(date_time_before5min, "%Y-%m-%d %H:%M:%S"), datetime.strftime(date_time_before10min, "%Y-%m-%d %H:%M:%S"), datetime.strftime(date_time_before15min, "%Y-%m-%d %H:%M:%S")
    logger.debug(min_id)
    logger.debug(min_id_before5min)
    logger.debug(min_id_before10min)
    return [min_id, min_id_before5min, min_id_before10min]


# 读取历史异常结果
def read_abn_result_by_restype(resource_type, month_time, min_id_before5min, min_id_before10min):
    '''
    读取异常结果，上个5min和上上个5min
    @resource_type: AMF | SMF | UPF | SWITCH | INTERFACE
    @data_dt: "202108251005"
    '''
    # 取index和cols
    input_idx_name = ES_INDEX_CFG["output_index_5gc"]["window"]
    input_idx_name = input_idx_name.replace("{YYYYMM}", month_time)
    col_list = ES_INDEX_CFG["abn_cols"]["5gc"]

    logger.debug(input_idx_name)
    query = {
        "track_total_hits": "true",
        "query": {
            "bool": {
                "filter": {
                    "range": {
                        "minId": {
                            "gte": min_id_before10min,
                            "lte": min_id_before5min
                        }
                    }
                },
                "must": [
                    {
                        "match": {
                            "resourceType": resource_type
                        }
                    }
                ]
            }
        }
    }
    logger.debug(query)
    params = {
        "_source": ",".join(col_list),
        "size": ES_SCROLL_ROW_NUM
    }
    filter_path = ["hits.total.value", "hits.hits._source", "_scroll_id"]
    kwargs = {"index": input_idx_name, "body": query, "params": params, "filter_path": filter_path}
    rows = search_with_scroll(tag="new", **kwargs)
    data = [p["_source"] for p in rows]
    if len(data) > 0:
        nf_df = pd.DataFrame(data)
    else:
        nf_df = pd.DataFrame(columns=col_list)
    return nf_df


# 写入数据
def write_es(idx_name: str, output_df: pd.DataFrame, id_cols=None, mappings=None):
    """
    向ES写入数据。
    @idx_name: 索引名称
    @output_df: 写es的dataframe
    @id_cols: 构建es中唯一键的列
    @mappings: es列mapping
    """
    logger.info(f"写入数据到: {idx_name}")
    create_index(idx_name, mappings)

    if len(output_df) == 0:
        logger.info("待写入的数据帧不包含数据，跳过ES写入")
        return

    ## dataframe预处理
    result = output_df.to_json(orient="records")
    parsed = json.loads(result)
    output_df_len = len(output_df)
    ## 批量写入
    action = ({
        "_index": idx_name,
        "_source": record,
    } for record in parsed)
    ret = bulk_to_es(action)
    if len(output_df) == ret[0]:
        logger.info(f"数据写入成功，条数: {output_df_len}")
    else:
        logger.error("数据写入异常")



def read_nf(resource_type: str, min_id_before5min: str, min_id_before10min: str):
    '''
    读取上次预测的全量网元数据
    @resource_type: AMF | SMF | UPF | SWITCH | INTERFACE
    @data_dt: "202108251005"
    '''
    input_idx_name: str = ES_INDEX_CFG["output_index_5gc"]["future6h"]
    input_idx_name_first, input_idx_name_second = input_idx_name.replace("{YYYYMMDD}", time.strftime("%Y%m%d", time.strptime(min_id_before5min, "%Y-%m-%d %H:%M:%S"))), input_idx_name.replace("{YYYYMMDD}", time.strftime("%Y%m%d", time.strptime(min_id_before10min, "%Y-%m-%d %H:%M:%S")))
    if input_idx_name_first == input_idx_name_second:
        input_idx_name = input_idx_name_first
    else:
        input_idx_name = input_idx_name_first + "," + input_idx_name_second
    cols = ES_INDEX_CFG["input_ne_cols_5gc"]
    ## 取index
    logger.info(input_idx_name)
    ## 初始化query
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "minId": {
                                "gte": min_id_before10min,
                                "lte": min_id_before5min
                            }
                        }
                    },
                    {
                        "match": {
                            "resourceType": resource_type
                        }
                    }
                ]
            }
        },
        "aggs": {
            "f1": {
                "terms": {
                    ## 将要剔重的字段拼接起来
                    "script": " + '|' + ".join([f"doc['{col}'].value" for col in cols]),
                    "size": ES_SCROLL_ROW_NUM
                }
            },
        }
    }
    if resource_type == "INTERFACE":
        query["query"]["bool"]["must_not"] = [{
            "match": {
                "itemPara": "-1"
            }
        }]
    else:
        query["query"]["bool"]["must"].append({
            "match": {
                "itemPara": "-1"
            }
        })
    logger.debug(query)
    params = {
        "size": ES_SCROLL_ROW_NUM
    }
    filter_path = ["aggregations.f1.buckets"]
    es = _init_es_conn_pool()
    res = es.search(index=input_idx_name, body=query, params=params, filter_path=filter_path, ignore=404)
    logger.debug(res)
    if res.get("error") is not None:
        if int(res['status']) != 404:
            logger.debug(res)
            logger.debug(f"查询ES发生错误: {res['status']}")
        return pd.DataFrame(columns=cols)
    results = res["aggregations"]["f1"]["buckets"]
    logger.debug(results)
    rows = list(t["key"].split("|") for t in results)
    if len(rows) > 0:
        nf_df = pd.DataFrame.from_dict(data=rows)
        nf_df.columns = cols
    else:
        nf_df = pd.DataFrame(columns=cols)
    logger.debug(nf_df)
    return nf_df


