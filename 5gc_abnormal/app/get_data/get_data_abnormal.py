from loguru import logger
from app.db.es_redis_config import ES_INDEX_CFG
from app.db.es_redis_config import ES_OUT_INDEX_MAPPING_5GC_VERT, ES_OUT_INDEX_MAPPING_5GC_WINDOW
from app.model.abnormal import run_abnormal_5gc
from app.db.elasticsearchwzh import create_index, bulk_to_es, get_min_id
import json
from hashlib import md5
import time

# 异常检测主函数
def model_abnormal(*, resource_type, input_idx_name):
    '''
    按网元类型批量执行输出网元异常及窗口异常。
    '''
    logger.info('开始异常检测')
    min_id, min_id_before5min, min_id_before10min = get_min_id()
    month_time = time.strftime("%Y%m", time.strptime(min_id, "%Y-%m-%d %H:%M:%S"))
    result_df, window_df = run_abnormal_5gc(resource_type, input_idx_name, min_id, min_id_before5min, min_id_before10min, month_time)
    idx_name_vert = ES_INDEX_CFG['output_index_5gc']['vert'].replace('{YYYYMM}', month_time)
    idx_name_window = ES_INDEX_CFG['output_index_5gc']['window'].replace('{YYYYMM}', month_time)
    write_es(idx_name_vert, result_df, id_cols=ES_INDEX_CFG['output_cols_5gc_unique']['vert'], mappings=ES_OUT_INDEX_MAPPING_5GC_VERT)
    write_es(idx_name_window, window_df, id_cols=ES_INDEX_CFG['output_cols_5gc_unique']['window'], mappings=ES_OUT_INDEX_MAPPING_5GC_WINDOW)
    del result_df
    del window_df


def write_es(idx_name: str, output_df, id_cols=None, mappings=None):
    """
    向ES写入数据。
    @idx_name: 索引名称
    @output_df: 写es的dataframe
    @id_cols: 构建es中唯一键的列
    @mappings: es列mapping
    """
    logger.info(f"写入数据到: {idx_name}")
    result = output_df.to_json(orient="records")
    parsed = json.loads(result)
    create_index(idx_name, mappings)
    actions = ({
        "_index": idx_name,
        "_source": record,
        "_id": get_record_hash(record, id_cols)
    } for record in parsed)
    ret = bulk_to_es(actions)
    output_df_len = len(output_df)
    if len(output_df) == ret[0]:
        logger.info(f"数据写入成功，条数: {output_df_len}")
    else:
        logger.error("数据写入异常")


def get_record_hash(record, id_cols=None):
    """
    输出唯一索引，同样的数据不会在ES存多份。
    @record 行记录
    @id_cols 构建es中唯一键的列
    """
    if id_cols is None:
        id_cols = record.keys()
    li = []
    for col in id_cols:
        val = record.get(col)
        if val is None:
            val = ""
        li.append(str(val))
    return md5(("-".join(li)).encode("utf-8")).hexdigest()