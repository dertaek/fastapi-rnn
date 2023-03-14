"""
修改预测以及异常检测模块参数文件，方便直接修改已经部署的容器内的预测以及异常检测参数。需要对参数文件挂载vol数据卷做持久化。
"""
from fastapi import APIRouter
from app.api.routes.schemas import ChangePredictJobConfig, ResponsePredictJobConfig, ChangeAbnormalJobConfig, ResponseAbnormalJobConfig
from loguru import logger
from app.get_data.post_predict_request import yaml_decode, yaml_save


router = APIRouter()

# 修改预测参数模块
@router.post("/modify_predict_config", response_model=ResponsePredictJobConfig)
async def ChangePredictConfig(req: ChangePredictJobConfig):
    changed_config = req.config_names
    job_config = yaml_decode("app/vol", "job_config.yml")["SubmitPredictJobRequest"]
    data = {"SubmitPredictJobRequest": {"resource_type": req.resource_type if "resource_type" in changed_config else
                                        job_config["resource_type"],
                                        "predict_method": req.predict_method.value if "predict_method" in changed_config else
                                        job_config["predict_method"],
                                        "predict_history": req.predict_history if "predict_history" in changed_config else
                                        job_config["predict_history"],
                                        "predict_sleep_time": req.predict_sleep_time if "predict_sleep_time" in changed_config else
                                        job_config["predict_sleep_time"]}}
    yaml_save(data, "app/vol", "job_config.yml")
    logger.info("预测参数修改成功！")
    return data["SubmitPredictJobRequest"]


# 修改异常检测参数模块
@router.post("/modify_abnormal_config", response_model=ResponseAbnormalJobConfig)
async def ChangeAbnormalConfig(req: ChangeAbnormalJobConfig):
    changed_config = req.config_names
    job_config = yaml_decode("app/vol", "job_config.yml")["SubmitAbnormalJobRequest"]
    data = {"SubmitPredictJobRequest": {"resource_type": req.resource_type if "resource_type" in changed_config else
                                        job_config["resource_type"],
                                        "abnormal_sleep_time": req.resource_type if "abnormal_sleep_time" in changed_config else
                                        job_config["abnormal_sleep_time"]}}
    yaml_save(data, "app/vol", "job_config.yml")
    logger.info("异常检测参数修改成功！")
    return data["SubmitAbnormalJobRequest"]
