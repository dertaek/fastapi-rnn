import socket
from typing import List
import itertools
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings, Secret


API_PREFIX = "/api"

VERSION = "1.1.0"

# 获取本机地址
HOST_NAME = socket.gethostbyname(socket.gethostname())

## 从环境变量读取参数配置
config = Config(".env.conf")

ENV: bool = config("ENV", default="release")

DEBUG: bool = config("DEBUG", cast=bool, default=False)
PROJECT_NAME: str = config("PROJECT_NAME", default="5gc application")

## 私钥本质只是一个不会再traceback暴露的字符串
SECRET_KEY: Secret = config("SECRET_KEY", cast=Secret, default="***************")
ALLOWED_HOSTS: List[str] = config(
    "ALLOWED_HOSTS",
    cast=CommaSeparatedStrings,
    default="",
)

## ES配置
ES_TIMEOUT: int = config("ES_TIMEOUT", cast=int, default=60)
ES_USERNAME: str = config("ES_USERNAME", default="*********************")
ES_PASSWORD: str = config("ES_PASSWORD", default="*********************")
# es主机地址
ES_HOSTS: List[str] = config(
    "ES_HOSTS",
    cast=CommaSeparatedStrings,
    default="****************,*****************,****************"
)
ES_RETRY_TIMES: int = config("ES_RETRY_TIMES", cast=int, default=3)
# es游标查询设置
ES_SCROLL_ROW_NUM: int = config("ES_SCROLL_ROW_NUM", cast=int, default=5000)

## Redis配置
REDIS_HOST: str = config("REDIS_HOST", default="localhost")
REDIS_HOSTS: List[str] = config(
    "REDIS_HOSTS",
    cast=CommaSeparatedStrings,
    default="******************,***************,**************,
)
REDIS_PORT: int = config("REDIS_PORT", cast=int, default=6379)
REDIS_PORTS: List[str] = config(
    "REDIS_PORTS",
    cast=CommaSeparatedStrings,
    default="31046,31046,31046,31046"
)
REDIS_USERNAME: str = config("REDIS_USERNAME", default="**************")
REDIS_PASSWORD: str = config("REDIS_PASSWORD", default="*************")
REDIS_DB: int = config("REDIS_DB", cast=int, default=17)

## REDIS集群节点拼接
hosts = zip(itertools.repeat("host"), REDIS_HOSTS)
ports = zip(itertools.repeat("port"), REDIS_PORTS)
nodes = zip(hosts, ports)
REDIS_CLUSTER = [dict(node) for node in nodes]
## auth拼接
REDIS_AUTH = REDIS_USERNAME + "#" + REDIS_PASSWORD

## SPARK
SPARK_DRIVER_MEM: str = config("SPARK_DRIVER_MEM", default="2g")
SPARK_EXECUTOR_MEM: str = config("SPARK_EXECUTOR_MEM", default="2g")
SPARK_PARALLELISM: str = config("SPARK_PARALLELISM", default="50")

# 权重df的字段
WEIGHT_COLS_NEED = ["resourceType", "kpiId", "abnScoreWeight", "isRateKpi"]