"""
接收5gc_producer请求模块
"""
from fastapi import APIRouter, BackgroundTasks
from loguru import logger
from app.api.routes.schemas import RNNConfigModify, ResponseRNNConfig, PredictDataConfig, AbnormalConfig
from app.get_data.get_data_predict import yaml_decode, yaml_save, volPath, get_data_predict
from app.get_data.get_data_abnormal import model_abnormal


router = APIRouter()


# 接收预测请求，添加请求进入后台任务
@router.post("/predict")
def run_predict(req: PredictDataConfig, background_tasks: BackgroundTasks):
    background_tasks.add_task(get_data_predict, resource_type=req.type_.value, resid=req.res_id, input_idx_name=req.input_idx_name,
                              predict_method=req.method.value)
    return "predict request posted!!"


# 接收异常检测请求，添加请求进入后台任务
@router.post("/abnormal")
def run_abnormal(req: AbnormalConfig, background_tasks: BackgroundTasks):
    background_tasks.add_task(model_abnormal, resource_type=req.type_.value, input_idx_name=req.input_idx_name)
    return "abnormal request posted!!"


# 提交异常检测RNN参数修改请求
@router.post("/modify_RNN_config", response_model=ResponseRNNConfig)
async def modify_rnn_config(config: RNNConfigModify):
    data = None
    rnn_config = yaml_decode(volPath, "RNN.yml")
    if config.type_ == 'AMF':
        data = {"AMF": {
            "hidden_size": config.hidden_size if "hidden_size" in config.config_names else rnn_config["AMF"]["hidden_size"],
            "num_layers": config.num_layers if "num_layers" in config.config_names else rnn_config["AMF"]["num_layers"],
            "nonlinearity": config.nonlinearity.value if "nonlinearity" in config.config_names else rnn_config["AMF"]["nonlinearity"],
            "bias": config.bias if "bias" in config.config_names else rnn_config["AMF"]["bias"],
            "batch_first": config.batch_first if "batch_first" in config.config_names else rnn_config["AMF"]["batch_first"],
            "dropout": config.dropout if "dropout" in config.config_names else rnn_config["AMF"]["dropout"],
            "bidirectional": config.bidirectional if "bidirectional" in config.config_names else rnn_config["AMF"]["bidirectional"],
            "learning_rate": config.learning_rate if "learning_rate" in config.config_names else rnn_config["AMF"]["learning_rate"]},
            "SMF": rnn_config['SMF'],
            "UPF": rnn_config['UPF'],
            "UDM": rnn_config['UDM'],
            "SWITCH": rnn_config['SWITCH'],
            "INTERFACE": rnn_config['INTERFACE']}

    elif config.type_ == 'SMF':
        data = {"SMF": {
            "hidden_size": config.hidden_size if "hidden_size" in config.config_names else rnn_config["SMF"][
                "hidden_size"],
            "num_layers": config.num_layers if "num_layers" in config.config_names else rnn_config["SMF"]["num_layers"],
            "nonlinearity": config.nonlinearity.value if "nonlinearity" in config.config_names else rnn_config["SMF"][
                "nonlinearity"], "bias": config.bias if "bias" in config.config_names else rnn_config["SMF"]["bias"],
            "batch_first": config.batch_first if "batch_first" in config.config_names else rnn_config["SMF"][
                "batch_first"],
            "dropout": config.dropout if "dropout" in config.config_names else rnn_config["SMF"]["dropout"],
            "bidirectional": config.bidirectional if "bidirectional" in config.config_names else rnn_config["SMF"][
                "bidirectional"],
            "learning_rate": config.learning_rate if "learning_rate" in config.config_names else rnn_config["SMF"][
                "learning_rate"]},
            "AMF": rnn_config['AMF'],
            "UPF": rnn_config['UPF'],
            "UDM": rnn_config['UDM'],
            "SWITCH": rnn_config['SWITCH'],
            "INTERFACE": rnn_config['INTERFACE']}

    elif config.type_ == 'UPF':
        data = {"UPF": {
            "hidden_size": config.hidden_size if "hidden_size" in config.config_names else rnn_config["UPF"][
                "hidden_size"],
            "num_layers": config.num_layers if "num_layers" in config.config_names else rnn_config["UPF"]["num_layers"],
            "nonlinearity": config.nonlinearity.value if "nonlinearity" in config.config_names else rnn_config["UPF"][
                "nonlinearity"], "bias": config.bias if "bias" in config.config_names else rnn_config["UPF"]["bias"],
            "batch_first": config.batch_first if "batch_first" in config.config_names else rnn_config["UPF"][
                "batch_first"],
            "dropout": config.dropout if "dropout" in config.config_names else rnn_config["UPF"]["dropout"],
            "bidirectional": config.bidirectional if "bidirectional" in config.config_names else rnn_config["UPF"][
                "bidirectional"],
            "learning_rate": config.learning_rate if "learning_rate" in config.config_names else rnn_config["UPF"][
                "learning_rate"]},
            "SMF": rnn_config['SMF'],
            "AMF": rnn_config['AMF'],
            "UDM": rnn_config['UDM'],
            "SWITCH": rnn_config['SWITCH'],
            "INTERFACE": rnn_config['INTERFACE']}

    elif config.type_ == 'UDM':
        data = {"UDM": {
            "hidden_size": config.hidden_size if "hidden_size" in config.config_names else rnn_config["UDM"][
                "hidden_size"],
            "num_layers": config.num_layers if "num_layers" in config.config_names else rnn_config["UDM"]["num_layers"],
            "nonlinearity": config.nonlinearity.value if "nonlinearity" in config.config_names else rnn_config["UDM"][
                "nonlinearity"], "bias": config.bias if "bias" in config.config_names else rnn_config["UDM"]["bias"],
            "batch_first": config.batch_first if "batch_first" in config.config_names else rnn_config["UDM"][
                "batch_first"],
            "dropout": config.dropout if "dropout" in config.config_names else rnn_config["UDM"]["dropout"],
            "bidirectional": config.bidirectional if "bidirectional" in config.config_names else rnn_config["UDM"][
                "bidirectional"],
            "learning_rate": config.learning_rate if "learning_rate" in config.config_names else rnn_config["UDM"][
                "learning_rate"]},
            "SMF": rnn_config['SMF'],
            "UPF": rnn_config['UPF'],
            "AMF": rnn_config['AMF'],
            "SWITCH": rnn_config['SWITCH'],
            "INTERFACE": rnn_config['INTERFACE']}

    elif config.type_ == 'SWITCH':
        data = {"SWITCH": {
            "hidden_size": config.hidden_size if "hidden_size" in config.config_names else rnn_config["SWITCH"][
                "hidden_size"],
            "num_layers": config.num_layers if "num_layers" in config.config_names else rnn_config["SWITCH"][
                "num_layers"],
            "nonlinearity": config.nonlinearity.value if "nonlinearity" in config.config_names else
            rnn_config["SWITCH"]["nonlinearity"],
            "bias": config.bias if "bias" in config.config_names else rnn_config["SWITCH"]["bias"],
            "batch_first": config.batch_first if "batch_first" in config.config_names else rnn_config["SWITCH"][
                "batch_first"],
            "dropout": config.dropout if "dropout" in config.config_names else rnn_config["SWITCH"]["dropout"],
            "bidirectional": config.bidirectional if "bidirectional" in config.config_names else rnn_config["SWITCH"][
                "bidirectional"],
            "learning_rate": config.learning_rate if "learning_rate" in config.config_names else rnn_config["SWITCH"][
                "learning_rate"]},
            "SMF": rnn_config['SMF'],
            "UPF": rnn_config['UPF'],
            "UDM": rnn_config['UDM'],
            "AMF": rnn_config['AMF'],
            "INTERFACE": rnn_config['INTERFACE']}

    elif config.type_ == 'INTERFACE':
        data = {"INTERFACE": {
            "hidden_size": config.hidden_size if "hidden_size" in config.config_names else rnn_config["INTERFACE"][
                "hidden_size"],
            "num_layers": config.num_layers if "num_layers" in config.config_names else rnn_config["INTERFACE"][
                "num_layers"],
            "nonlinearity": config.nonlinearity.value if "nonlinearity" in config.config_names else
            rnn_config["INTERFACE"]["nonlinearity"],
            "bias": config.bias if "bias" in config.config_names else rnn_config["INTERFACE"]["bias"],
            "batch_first": config.batch_first if "batch_first" in config.config_names else rnn_config["INTERFACE"][
                "batch_first"],
            "dropout": config.dropout if "dropout" in config.config_names else rnn_config["INTERFACE"]["dropout"],
            "bidirectional": config.bidirectional if "bidirectional" in config.config_names else
            rnn_config["INTERFACE"]["bidirectional"],
            "learning_rate": config.learning_rate if "learning_rate" in config.config_names else
            rnn_config["INTERFACE"]["learning_rate"]},
            "SMF": rnn_config['SMF'],
            "UPF": rnn_config['UPF'],
            "UDM": rnn_config['UDM'],
            "SWITCH": rnn_config['SWITCH'],
            "AMF": rnn_config['AMF']}

    yaml_save(data, volPath, "RNN.yml")
    logger.info("rnn参数修改成功！")
    return data

