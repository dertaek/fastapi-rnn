# import json
# import os
# from loguru import logger
# import pandas as pd
# from cacheout import Cache
# # from app.core.config import CACHE_PATH, LOG_PATH, PREFIX_NE, HOST_NAME
import os

import pandas as pd
#
#
# # 全局缓存
# cache = Cache(maxsize=1024)
#
#
# ################################################################
# ## 内存缓存读写
# ################################################################
# # 保存变量，且本地保存文件
# def cache_set(key, val: object):
#     cache.set(key, val)
#     try:
#         fo = open(CACHE_PATH + HOST_NAME + "_" + key, "w")
#         fo.write(json.dumps(val))
#         fo.close()
#     except IOError as e:
#         logger.error(e)
#
#
# # 取变量返回原始类型
# def cache_get(key):
#     val = cache.get(key)
#     if val is None:
#         try:
#             fo = open(CACHE_PATH + HOST_NAME + "_" + key, "r")
#             val = fo.read()
#             fo.close()
#             val = json.loads(val)
#             cache.set(key, val)
#         except json.decoder.JSONDecodeError as je:
#             logger.error(je)
#             return
#         except IOError as e:
#             logger.error(e)
#             return
#     return val
#
#
# # 保存变量为CSV，且本地保存文件
# def cache_set_dataframe(key, df: pd.DataFrame):
#     cache.set(key, df)
#     try:
#         df.to_csv(CACHE_PATH + HOST_NAME + "_" + key)
#     except IOError as e:
#         logger.error(e)
#
#
# # 往缓存中追加内容
# def cache_set_dataframe_additive(key, df: pd.DataFrame):
#     val = cache_get_dataframe(key)
#     if val is None:
#         val = df
#     else:
#         val = pd.concat([val, df])
#     try:
#         df.to_csv(CACHE_PATH + HOST_NAME + "_" + key)
#         # df.to_csv(CACHE_PATH + key, mode="a", header=False)
#     except IOError as e:
#         logger.error(e)
#
#
# # 取变量返回dataframe
# def cache_get_dataframe(key) -> pd.DataFrame:
#     val = cache.get(key)
#     if val is None:
#         try:
#             val = pd.read_csv(CACHE_PATH + HOST_NAME + "_" + key, index_col=0)
#         except IOError as e:
#             logger.error(e)
#     return val
#
#
# ################################################################
# ## 载入权重配置
# ################################################################
#
# # 目录创建
# def create_dir_and_cp_weight_cfg():
#     try:
#         os.makedirs(LOG_PATH)
#     except OSError as e:
#         logger.error(e)
#     try:
#         os.makedirs(CACHE_PATH)
#     except OSError as e:
#         logger.error(e)
#
#
# create_dir_and_cp_weight_cfg()
# i_kpi_weight = cache.get("I_DF_KPI_WEIGHT")
# if i_kpi_weight is None:
#     try:
#         i_kpi_weight = pd.read_csv("app/resources/" + "I_DF_KPI_WEIGHT", index_col=0)
#     except IOError as e:
#         logger.error(e)
# if i_kpi_weight is None:
#     raise Exception('权重配置文件不存在，请检查')
#
#
def kpi_weight_read_cache(resource_type):
    i_kpi_weight = pd.read_csv("app/resource/" + "I_DF_KPI_WEIGHT", index_col=0)
    # i_kpi_weight = pd.read_csv(os.path.abspath(".") + "/I_DF_KPI_WEIGHT", index_col=0)
    return i_kpi_weight[i_kpi_weight['resourceType'] == resource_type]
#
#
# ################################################################
# ## 读取和写入全量网元
# ################################################################
# def all_ne_read_cache(resource_type, min_id=None):
#     df_ne = cache_get_dataframe(PREFIX_NE + resource_type.upper())
#     if df_ne is None:
#         logger.info('当前不存在类型为[' + resource_type + ']的全量网元信息，重新查询')
#     return df_ne
#
#
# def all_ne_save_cache(resource_type, df):
#     # 先取再合并，最后写入
#     df_ne = cache_get_dataframe(PREFIX_NE + resource_type.upper())
#     df_final = pd.concat([df_ne, df]).drop_duplicates()
#     cache_set_dataframe(PREFIX_NE + resource_type.upper(), df_final)