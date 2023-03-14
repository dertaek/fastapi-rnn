# from datetime import datetime
# from hashlib import md5
# import copy
# from app.libs.util import format_timestr
# from app.libs.cache import cache_get, cache_set
#
# # 任务信息
# g_job_info: dict = {}
# g_job_hash: dict = {}
# g_job_id_seq: int = 0
#
#
# def init_job_info():
#     '''
#     初始化任务信息，读取或重置
#     '''
#     global g_job_info, g_job_hash, g_job_id_seq
#     g_job_info = cache_get("G_JOB_INFO")
#     g_job_hash = cache_get("G_JOB_HASH")
#     g_job_id_seq = cache_get("G_JOB_ID_SEQ")
#     if g_job_info is None or g_job_hash is None or g_job_id_seq is None:
#         reset_job_info()
#
#
# def reset_job_info():
#     '''
#     重置任务信息
#     '''
#     global g_job_info, g_job_hash, g_job_id_seq
#     # 任务执行信息
#     g_job_info = {}
#     # 参数hash唯一，保证正在运行的任务唯一性
#     g_job_hash = {}
#     # 参数自增
#     g_job_id_seq = 0
#     job_cache()
#
#
# # 校验
# def job_get_hash(info: dict):
#     job_info = copy.deepcopy(info)
#     del job_info["force_run"]
#     job_str = ",".join(str(val) for val in job_info.values())
#     return md5(job_str.encode(encoding="utf-8")).hexdigest()
#
#
# # 查看Job信息
# def job_get_info_by_id(job_id=None):
#     global g_job_info
#     if job_id is None:
#         return g_job_info
#     else:
#         return g_job_info.get(job_id)
#
#
# # 当前任务是否正在运行
# def job_is_running(info: dict):
#     global g_job_hash
#     hash_id = job_get_hash(info)
#     hash_info = g_job_hash.get(hash_id)
#     if hash_info is not None:
#         return True, hash_info
#     else:
#         return False, None
#
#
# # 根据job_id查状态
# def job_get_by_id(job_id):
#     info = g_job_info.get(job_id)
#     if info is not None:
#         return info
#
#
# # 根据job_id查状态
# def job_get_all():
#     global g_job_info
#     return g_job_info
#
#
# # 根据job_id查状态
# def job_get_state_by_id(job_id):
#     global g_job_info
#     info = g_job_info.get(job_id)
#     if info is not None:
#         # return info["state"]
#         return info.state
#
#
# # 提交新任务
# def job_ensure(job_type: str, info: dict):
#     hash_id = job_get_hash(info)
#     global g_job_id_seq, g_job_info, g_job_hash
#     g_job_id_seq = g_job_id_seq + 1
#     g_job_info[g_job_id_seq] = {
#         "job_type": job_type,
#         "state": "RUNNING",
#         "info": info,
#         "execute_time": format_timestr(datetime.now(), 1),
#         "hash": hash_id
#     }
#     g_job_hash[hash_id] = g_job_id_seq
#     job_cache()
#     return g_job_id_seq
#
#
# # 完成job，更新信息
# def job_finish(job_id):
#     global g_job_info, g_job_hash
#     info = g_job_info.get(job_id)
#     info["state"] = "FINISH"
#     info["finish_time"] = format_timestr(datetime.now(), 1)
#     del g_job_hash[info["hash"]]
#     job_cache()
#
#
# # job出错，更新信息
# def job_error(job_id, exceptions=None):
#     global g_job_info, g_job_hash
#     info = g_job_info.get(job_id)
#     info["state"] = "ERROR"
#     info["finish_time"] = format_timestr(datetime.now(), 1)
#     if exceptions is not None:
#         info["exceptions"] = str(exceptions)
#     del g_job_hash[info["hash"]]
#     job_cache()
#
#
# # 保存任务信息
# def job_cache():
#     global g_job_info, g_job_hash, g_job_id_seq
#     cache_set("G_JOB_INFO", g_job_info)
#     cache_set("G_JOB_HASH", g_job_hash)
#     cache_set("G_JOB_ID_SEQ", g_job_id_seq)
