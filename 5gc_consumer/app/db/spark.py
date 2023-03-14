from pyspark.sql import SparkSession
from app.api.config.config import SPARK_DRIVER_MEM, SPARK_EXECUTOR_MEM, SPARK_PARALLELISM

g_spark: SparkSession = None


def init_or_get_spark():
    """
    初始化或获取Spark。
    """
    global g_spark
    if g_spark is None:
        g_spark = (
            SparkSession.builder
            .config("spark.driver.memory", SPARK_DRIVER_MEM)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEM)
            .config("spark.default.parallelism", SPARK_PARALLELISM)
            .getOrCreate()
        )
    return g_spark


def stop_spark():
    """
    停止spark。
    """
    global g_spark
    g_spark.stop()
