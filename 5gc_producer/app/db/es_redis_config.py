"""
es参数配置文件
"""
import itertools
from typing import List
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config(".env.conf")

# ENV: bool = config("ENV", default="release")
ENV: bool = config("ENV", default="dev")
## ES配置
ES_TIMEOUT: int = config("ES_TIMEOUT", cast=int, default=60)
ES_USERNAME: str = config("ES_USERNAME", default="************")
ES_PASSWORD: str = config("ES_PASSWORD", default="************)
# es主机地址
ES_HOSTS: List[str] = config(
    "ES_HOSTS",
    cast=CommaSeparatedStrings,
    default="http://**********************,http://*******************,http://********************"
)
ES_RETRY_TIMES: int = config("ES_RETRY_TIMES", cast=int, default=3)
# es游标查询设置
ES_SCROLL_ROW_NUM: int = config("ES_SCROLL_ROW_NUM", cast=int, default=500)
# es游标scroll过期时间
ES_SCROLL_TIMER: str = config("ES_SCROLL_TIMER", default="15m")

# 输入索引中各类型的设备
ES_DEVICE_SETS = {
    "AMF": ["DEV_5G_A_ER_COM", "DEV_5G_A_HW_COM", "DEV_5G_A_ZX_COM"],
    "SMF": ["DEV_5G_S_ER_COM", "DEV_5G_S_HW_COM", "DEV_5G_S_ZX_COM"],
    "UPF": ["DEV_5G_U_ER_COM", "DEV_5G_U_HW_COM", "DEV_5G_U_ZX_COM", "DEV_5G_U_YJY_COM"],
    "UDM": ["DEV_5G_H_HW_COM", "DEV_5G_H_ER_COM", "DEV_5G_H_ZX_COM"],
    "SWITCH": ["DEV_DC_T_HW_COM", "DEV_DC_T_ZX_COM", "DEV_DC_T_H3_COM", "DEV_DC_E_HW_COM", "DEV_DC_E_ZX_COM", "DEV_DC_E_H3_COM"],
    "INTERFACE": ["DEV_DC_T_HW_COM", "DEV_DC_T_ZX_COM", "DEV_DC_T_H3_COM", "DEV_DC_E_HW_COM", "DEV_DC_E_ZX_COM", "DEV_DC_E_H3_COM"]
}

# 网元类型对指标的映射关系
ES_KPI_SET = {
    "AMF": ["5gcAmfAuthSuccessRate", "5gcAmfAttSucRateWithOutUser", "5gcAmfN2PageReqRespNum", "5gcAmfSwit5gTo4gSuccRate", "5gcAmfNet4gTo5gReqSuccRate", "5gcAmfNet5gTo4gSuccRate", "5gcAmfXnInterSwitSuccRate", "5gcAmfN2InterSwitSuccRate", "5gcAmfInterSwitInSuccRate", "5gcAmfInterSwitOutSuccRate", "5gcAmfAvgSysLoad"],
    "SMF": ["5gcSmfDnn5GPduSessNum", "5gcSmfSnssaiAvg5gNum", "5gcSmfDnn5GPduSessMaxNum", "5gcSmfSnssaiMax5gNum", "5gcSmfIMSEstSucRate", "5gcSmfSessBuiSuccRate", "5gcSmfFallbackSuccRate", "5gcSmfSysAvgLoad"],
    "UPF": ["5gcUpfPduSessSuccRate", "5gcUpfN3SaveGtp13MessThou", "5gcUpfN3SaveGtp11MessMaxKbRate", "5gcUpfN3ReceGtpMaxKapRateByDn", "5gcUpfN3ReceGtpMaxKbMessRate", "5gcUpfN3SendGtpMessThou", "5gcUpfN3SendGtpMaxKapRateByDn", "5gcUpfN3SendGtpMaxKbMessRate", "5gcUpfN3SnssaiSendMax1KapRate", "5gcUpfN3SnssaiSendMax2KapRate", "5gcUpfN6SendGtp13MessThouIpv4", "5gcUpfN6SendGtp13MessThouIpv6", "5gcUpfN6RecUsrPeakMaxKBRate", "5gcUpfN6SendGtp14MessThouIpv4", "5gcUpfN6SendGtp14MessThouIpv6", "5gcUpfN6UserSnssaiReceKapUserNum", "5gcUpfN6SendUsrPeakMaxKBRate", "5gcUpfOverThrouAvg", "5gcUpfForwSuccRate", "5gcUpfShaXinZhuFaSucceRate", "5gcUpfDownForwSuccRate", "5gcUpfXiaXinZhuFaSucceRate", "5gcUpfCpuUage", "5gcUpfSysAvgLoadCpu", "5gcUpfMemUsageRate", "5gcUpfSysAvgLoadMe"],
    "UDM": ["5gcUdmSubUserNum", "5gcUdm3gppActiUserNum", "5gcUdmAmfReqSuccRate", "5gcUdmReqMsgSuccRate", "5gcUdmSmfReqSuccRate", "5gcUdmSignSuccResqMessRate"],
    "SWITCH": ["Dcswitchcpuusage"],
    "INTERFACE": ["Dcswitchportincomingdiscardpackets", "Dcswitchportincomingerrorpackets", "Dcswitchportoutgoingdiscardpackets", "Dcswitchportoutgoingerrorpackets"]
}

## ES索引信息
ES_INDEX_CFG = {
    'input_index': 'pm_perf_itemrawinfo_{YYYYMMDD}',
    'input_cols': ['RESID', 'NODECODE', 'DEVICE_SET', 'ITEM_PARA', 'RECORD_TIME', 'ITEM_CODE', 'VALUE'],
    'input_cols_rename': ['resId', 'nodeCode', 'deviceSet', 'itemPara', 'recordTime', 'kpiId', 'kpiValue', 'resourceType', 'minId', 'vender'],
    'input_ne_cols_5gc': ['nodeCode', 'resId', 'vender', 'resourceType'],  # 全量网元剔重字段
    'input_ne_cols_5gy': ['nodeCode', 'itemPara', 'resId', 'vender', 'resourceType'],  # 全量网元剔重字段

    # 5gc输出索引名称
    'output_index_5gc': {
        'future6h': 'vgc_risk_control_forecast_kpi_{YYYYMM}',
        'vert': 'vgc_risk_control_kpi_{YYYYMM}',
        'window': 'vgc_risk_control_nf_{YYYYMM}'
    },

    # 5gc输出索引所有列
    'output_cols_5gc': {
        'future6h': ['nodeCode', 'reportTime', 'resId', 'resourceType', 'vender', 'kpiId', 'forecastValue', 'forecastValueUpper', 'forecastValueLower', 'cutId', 'createTime'],
        'vert': ['minId', 'nodeCode', 'resId', 'resourceType', 'vender', 'kpiId', 'kpiValue', 'forecastValue', 'forecastValueUpper', 'forecastValueLower', 'ifRangeFcst', 'abnLevel', 'abnScore', 'cutId', 'createTime'],
        'window': ['minId', 'nodeCode', 'resId', 'resourceType', 'vender', 'windowScore', 'isWindowAbn', 'isWindowAlarm', 'abnScore', 'abnLevel', 'trainingAlgo', 'cutId', 'abnRate', 'createTime']
    },
    # 5gc输出索引唯一键
    'output_cols_5gc_unique': {
        'future6h': ['minId', 'reportTime', 'resId', 'kpiId'],
        'vert': ['minId', 'resId', 'kpiId'],
        'window': ['minId', 'resId']
    },

    # 从异常输出索引中只取以下列
    'abn_cols': {
        '5gc': ['nodeCode', 'resId', 'vender', 'resourceType', 'abnScore'],
        '5gy': ['nodeCode', 'resId', 'itemPara', 'vender', 'resourceType', 'abnScore']
    },

    # 5gy输出索引名称
    'output_index_5gy': {
        'future6h': 'vgc_digicom_forecast_kpi_{YYYYMMDD}_new',
        'vert': 'vgc_digicom_kpi_{YYYYMMDD}',
        'abn': 'vgc_digicom_nf_abn_{YYYYMMDD}',
        'window': 'vgc_digicom_nf_window_{YYYYMMDD}',
    },
    # 5gy输出索引所有列
    'output_cols_5gy': {
        'future6h': ['minId', 'nodeCode', 'reportTime', 'resId', 'itemPara', 'resourceType', 'vender', 'kpiType', 'kpiTypeValue', 'kpiId', 'forecastValue', 'forecastValueUpper', 'forecastValueLower', 'createTime'],
        'vert': ['minId', 'nodeCode', 'resId', 'itemPara', 'resourceType', 'vender', 'kpiType', 'kpiTypeValue', 'kpiId', 'kpiValue', 'forecastValue', 'forecastValueUpper', 'forecastValueLower', 'ifRangeFcst', 'abnLevel', 'abnScore', 'createTime'],
        'abn': ['minId', 'nodeCode', 'resId', 'itemPara', 'resourceType', 'vender', 'abnScore', 'abnLevel', 'trainingAlgo', 'abnRate', 'createTime'],
        'window': ['minId', 'nodeCode', 'resId', 'itemPara', 'resourceType', 'vender', 'windowScore', 'isWindowAlarm', 'isWindowAbn', 'createTime']
    },
    # 5gy输出索引唯一键
    'output_cols_5gy_unique': {
        'future6h': ['minId', 'reportTime', 'resId', 'itemPara', 'kpiId'],
        'vert': ['minId', 'resId', 'itemPara', 'kpiId'],
        'abn': ['minId', 'resId', 'itemPara'],
        'window': ['minId', 'resId', 'itemPara']
    },
}

OUT_MAPPING = {
    "mappings": {
        "properties": {
            "RESID": {"type": "keyword"},
            "ITEM_PARA": {"type": "keyword"},
            "RECORD_TIME": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss"
            },
            "ITEM_CODE_VALUE": {"type": "keyword"},
            "CREATE_TIME": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss"
            }
        }
    }
}

OUT_ES_INDEX = {
    'input_index': 'pm_perf_itemrawinfo_{YYYYMMDD}'
}

## Redis配置
REDIS_HOST: str = config("REDIS_HOST", default="localhost")
REDIS_HOSTS: List[str] = config(
    "REDIS_HOSTS",
    cast=CommaSeparatedStrings,
    default="***************************",
)
REDIS_PORT: int = config("REDIS_PORT", cast=int, default=6379)
REDIS_PORTS: List[str] = config(
    "REDIS_PORTS",
    cast=CommaSeparatedStrings,
    default="**********************"
)
REDIS_USERNAME: str = config("REDIS_USERNAME", default="*********")
REDIS_PASSWORD: str = config("REDIS_PASSWORD", default="***********")
REDIS_DB: int = config("REDIS_DB", cast=int, default=17)

## REDIS集群节点拼接
hosts = zip(itertools.repeat("host"), REDIS_HOSTS)
ports = zip(itertools.repeat("port"), REDIS_PORTS)
nodes = zip(hosts, ports)
REDIS_CLUSTER = [dict(node) for node in nodes]
## auth拼接
REDIS_AUTH = REDIS_USERNAME + "#" + REDIS_PASSWORD