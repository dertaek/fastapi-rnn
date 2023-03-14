"""
获取各类型网元的resid，并提交请求给5gc_consumer
"""
from elasticsearch import Elasticsearch
from app.db.es_redis_config import ES_SCROLL_TIMER, ES_SCROLL_ROW_NUM, ES_INDEX_CFG
from loguru import logger
from datetime import datetime, timedelta, date

es = None


# 初始化es链接
def _init_es_conn_pool():
    global es
    if es is None or es.ping() is False:
        es_hosts = ["http://**************", "http://**************", "http://**********"]
        es_http_auth = ("****************", "***************")
        # es_hosts = ["http://127.0.0.1:9200"]
        # es = AsyncElasticsearch(es_hosts, timeout=60, max_retries=10, retry_on_timeout=True)
        es = Elasticsearch(es_hosts, http_auth=es_http_auth, timeout=60, max_retries=10, retry_on_timeout=True)
        logger.debug(f"建立ES_async连接: {es_hosts}")
    return es


# 翻页查询模块
def search_with_scroll(input_idx_name, query, params, filter_path):
    """
    分页查询操作封装
    """
    es = _init_es_conn_pool()
    res = es.search(index=input_idx_name, body=query, params=params, scroll=ES_SCROLL_TIMER,
                          filter_path=filter_path,
                          size=ES_SCROLL_ROW_NUM, ignore=404, timeout="60s")

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
        res_scroll = es.scroll(scroll_id=scroll_id, scroll=ES_SCROLL_TIMER, request_timeout=60)
        # await es.clear_scroll(scroll_id=scroll_id, request_timeout=60)
        logger.debug(f"分页查询，当前页数: {i + 1} 已加载")
        # results是列表嵌套字典
        results = results + res_scroll["hits"]["hits"]

    # 清除es.scroll上下文
    es.clear_scroll(scroll_id=scroll_id, request_timeout=60)
    return results


# 获取历史索引模块
def get_es_indices(current_datetime, predict_history=7, task_type="predict"):
    es = _init_es_conn_pool()
    idx_name = ES_INDEX_CFG["input_index"]
    if task_type == "predict":
        if type(current_datetime) == str:
            indices = [idx_name.replace("{YYYYMMDD}", (datetime.strptime(current_datetime, "%Y-%m-%d") +
                      timedelta(days=-d)).strftime('%Y%m%d')) for d in range(predict_history + 1)]
            result = [index for index in indices if es.indices.exists(index=index, request_timeout=60)]
        else:
            indices = [idx_name.replace("{YYYYMMDD}", (current_datetime + timedelta(days=-d)).strftime('%Y%m%d'))
                       for d in range(predict_history + 1)]
            result = [index for index in indices if es.indices.exists(index=index, request_timeout=60)]
        return ','.join(result)
    elif task_type == "abnormal":
        if type(current_datetime) == str:
            indices = [idx_name.replace("{YYYYMMDD}", (datetime.strptime(current_datetime, "%Y-%m-%d") +
                      timedelta(days=-d)).strftime('%Y%m%d')) for d in range(2)]
            result = [index for index in indices if es.indices.exists(index=index, request_timeout=60)]
        else:
            indices = [idx_name.replace("{YYYYMMDD}", (current_datetime + timedelta(days=-d)).strftime('%Y%m%d'))
                       for d in range(2)]
            result = [index for index in indices if es.indices.exists(index=index, request_timeout=60)]
        return ','.join(result)


# 获取网元列表模块
def get_resid_list(idx_name, dev_list, type_):
    now = datetime.now()
    after_time = now.strftime("%Y-%m-%d %H:%M:%S")
    before = now - timedelta(hours=3)
    before_time = before.strftime("%Y-%m-%d %H:%M:%S")
    query = {
        "size": 0,
        "query": {
            "bool": {
                "filter": {
                    "range": {
                        "RECORD_TIME": {
                            "gte": before_time,
                            "lte": after_time
                        }
                    }
                },
                "must": []
            }
        },
        "aggs": {
            "res_id": {
                "terms": {
                    "field": "RESID",
                    "size": 1500
                }
            }
        }
    }

    query["query"]["bool"]["must"].append({
        "terms": {
            "DEVICE_SET": dev_list
        }
    })

    if type_ == "INTERFACE":
        query["query"]["bool"]["must_not"] = [{
            "term": {
                "ITEM_PARA": "-1"
            }
        }]
    else:
        query["query"]["bool"]["must"].append({
            "term": {
                "ITEM_PARA": "-1"
            }
        })
    logger.debug(query)

    filter_path = ["aggregations"]

    es = _init_es_conn_pool()
    res = es.search(index=idx_name, body=query, filter_path=filter_path, ignore=404, timeout="60s")
    return res["aggregations"]["res_id"]["buckets"]