#######################################################
## Author: Nemo、llh
## Desc: 异常检测模块
#######################################################
import time

import pandas as pd
from loguru import logger
from pyspark.sql.types import StructType, StructField, StringType

from app.db.spark import init_or_get_spark
from app.db.es_redis_config import ES_INDEX_CFG
from app.api.config.config import WEIGHT_COLS_NEED
from app.libs.cache import kpi_weight_read_cache

from app.db.elasticsearchwzh import (
    read_input_by_restype,
    read_predict_result_by_restype,
    read_abn_result_by_restype
)

################################################################
## 异常检测，按{资源类型}批处理
################################################################
def run_abnormal_5gc(resource_type, input_idx_name, min_id, min_id_before5min, min_id_before10min, month_time):
    """
    5gc异常检测
    """
    pd.set_option('display.max_columns', None)
    day_time = time.strftime("%Y%m%d", time.strptime(min_id, "%Y-%m-%d %H:%M:%S"))
    # 上一次预测的结果(全量)
    futr6h_5gc = read_predict_result_by_restype(min_id)
    logger.debug(futr6h_5gc.head(5))

    # 当前5min的数据（按类型）
    vert_5gc = read_input_by_restype(resource_type, input_idx_name, min_id)
    vert_5gc.columns = ES_INDEX_CFG['input_cols_rename']
    logger.debug(vert_5gc.head(5))
    #
    # ## 指标权重配置
    feature_5gc = kpi_weight_read_cache(resource_type)
    feature_5gc = feature_5gc[WEIGHT_COLS_NEED]
    logger.debug(feature_5gc.head(5))
    #
    # # 获取异常网元结果表
    abnormal_5gc_last = read_abn_result_by_restype(resource_type, month_time, min_id_before5min, min_id_before10min) # 异常网元结果表逻辑系要弄清
    logger.debug(abnormal_5gc_last)

    # 取上一个5min预测的所有网元
    # nf_5gc = read_nf(resource_type, min_id_before5min, min_id_before10min)
    # logger.debug(nf_5gc.head(5))

    spark = init_or_get_spark()
    # 从es读取的预测结果表: ta_5gc_fcst_futr6h_day，时间: 上一个5分钟预测的当前5分钟的数据
    schema = StructType([StructField(col, StringType()) for col in futr6h_5gc.columns])
    futr6h_5gc_df = spark.createDataFrame(futr6h_5gc, schema=schema)
    futr6h_5gc_df.createOrReplaceTempView(f"ta_5gc_fcst_futr6h_day_{resource_type}_{day_time}")
    del futr6h_5gc
    del futr6h_5gc_df

    # 从es读取竖表数据t_5gc_perf_vert_5t: 当前5分钟的数据
    schema = StructType([StructField(col, StringType()) for col in vert_5gc.columns])
    vert_5gc_df = spark.createDataFrame(vert_5gc, schema=schema)
    vert_5gc_df.createOrReplaceTempView(f"t_5gc_perf_vert_5t_{resource_type}_{day_time}")
    del vert_5gc
    del vert_5gc_df

    # 从es读取特征表: dim_kpi_feature，5gc特征数据:
    schema = StructType([StructField(col, StringType()) for col in feature_5gc.columns])
    feature_5gc_df = spark.createDataFrame(feature_5gc, schema=schema)
    feature_5gc_df.createOrReplaceTempView(f"dim_kpi_feature_5gc_{resource_type}_{day_time}")
    logger.debug(feature_5gc_df)
    del feature_5gc
    del feature_5gc_df

    # 从es表读取前5分钟和10分钟的异常网元数据: ta_5gc_abnormal_day，前5分钟和10分钟
    schema = StructType([StructField(col, StringType()) for col in abnormal_5gc_last.columns])
    abnormal_5gc_last_df = spark.createDataFrame(abnormal_5gc_last, schema=schema)
    abnormal_5gc_last_df.createOrReplaceTempView(f"ta_5gc_abnormal_day_last_{resource_type}_{day_time}")
    del abnormal_5gc_last
    del abnormal_5gc_last_df

    # 从es读取全量网元
    # schema = StructType([StructField(col, StringType()) for col in nf_5gc.columns])
    # nf_5gc_df = spark.createDataFrame(nf_5gc, schema=schema)
    # nf_5gc_df.createOrReplaceTempView(f"dim_5gc_net_info_{resource_type}_{day_time}")

    # 关联操作
    result_5gc = spark.sql(f'''
        WITH tmp01 as
        (
            SELECT  DISTINCT t1.nodeCode
                   ,t1.resId
                   ,t1.kpiId
                   ,t2.kpiValue
                   ,cast(t1.forecastValue AS decimal(22, 2)) AS forecastValue
                   ,cast(t1.forecastValueUpper AS decimal(22, 2)) AS forecastValueUpper
                   ,cast(t1.forecastValueLower AS decimal(22, 2)) AS forecastValueLower
                   ,CASE WHEN (nvl(t2.kpiValue,'')='') or (cast(t2.kpiValue AS decimal(22,2)) >= cast(t1.forecastValueLower AS decimal(22,2)) AND cast(t2.kpiValue AS decimal(22,2)) <= cast(t1.forecastValueUpper AS decimal(22,2))) THEN 1  ELSE 0 END AS ifRangeFcst
                   ,CASE when nvl(t2.kpiValue,'') = '' then 0
                         when t1.kpiId IN ('5gcAmfNotUserReasonAuthSuccRate',
                                         '5gcAmfAttSucRateWithOutUser',
                                         '5gcAmfFinReqSuccRate',
                                         '5gcAmfN2SerReqSccRate',
                                         '5gcAmfSwit5gTo4gSuccRate',
                                         '5gcAmfSwit4gTo5gSuccRate',
                                         '5gcAmfNet4gTo5gReqSuccRate',
                                         '5gcAmfNet5gTo4gSuccRate',
                                         '5gcAmfXnInterSwitSuccRate1',
                                         '5gcAmfN2InterSwitSuccRate1',
                                         '5gcAmfInterSwitInSuccRate1',
                                         '5gcAmfInterSwitOutSuccRate1',
                                         '5gcSmfIMSEstSucRate',
                                         '5gcSmfSessBuiSuccRate',
                                         '5gcSmfFallbackSuccRate',
                                         '5gcUpfPduSessSuccRate',
                                         '5gcUpfXiaXinZhuFaSucceRate',
                                         '5gcUdmAmfSuccMessRate',
                                         '5gcUdmSmfSuccMessRate',
                                         '5gcUdmSignSuccMessRate',
                                         '5gcUpfShaXinZhuFaSucceRate',
                                         '5gcAmfMaxRegStatusUserNum',
                                         '5gcAmfPageReqNum',
                                         '5gcSmfAvgPduSessNum',
                                         '5gcSmfMaxPduSessNum',
                                         '5gcSmf4GAvgPduSessNum',
                                         '5gcSmf4GMaxPduSessNum',
                                         '5gcSmf45GAvgPduSessNum',
                                         '5gcSmf45GMaxPduSessNum',
                                         '5gcUpfUsrCurrOnlSessNum',
                                         '5gcUpfN3SaveGtp13MessThou',
                                         '5gcUpfN3ReceGtpMaxKbMessRate',
                                         '5gcUpfN3SendGtpMessThou',
                                         '5gcUpfN3SendGtpMaxKbMessRate',
                                         '5gcUpfN6SendGtp13MessThou',
                                         '5gcUpfN6RecUsrPeakMaxKBRate',
                                         '5gcUpfN6SendGtp14MessThou',
                                         '5gcUpfN6SendUsrPeakMaxKBRate',
                                         '5gcUpfN9RecUsrPeakMaxKBRate',
                                         '5gcUpfN9SendUsrPeakMaxKBRate',
                                         '5gcUpf7560Num',
                                         '5gcUdmSubUserNum',
                                         '5gcUdm3gppActiUserNum') and t2.kpiValue < 0.6 * t1.forecastValueLower then 100
                         when t1.kpiId IN ('5gcAmfNotUserReasonAuthSuccRate',
                                         '5gcAmfAttSucRateWithOutUser',
                                         '5gcAmfFinReqSuccRate',
                                         '5gcAmfN2SerReqSccRate',
                                         '5gcAmfSwit5gTo4gSuccRate',
                                         '5gcAmfSwit4gTo5gSuccRate',
                                         '5gcAmfNet4gTo5gReqSuccRate',
                                         '5gcAmfNet5gTo4gSuccRate',
                                         '5gcAmfXnInterSwitSuccRate1',
                                         '5gcAmfN2InterSwitSuccRate1',
                                         '5gcAmfInterSwitInSuccRate1',
                                         '5gcAmfInterSwitOutSuccRate1',
                                         '5gcSmfIMSEstSucRate',
                                         '5gcSmfSessBuiSuccRate',
                                         '5gcSmfFallbackSuccRate',
                                         '5gcUpfPduSessSuccRate',
                                         '5gcUpfXiaXinZhuFaSucceRate',
                                         '5gcUdmAmfSuccMessRate',
                                         '5gcUdmSmfSuccMessRate',
                                         '5gcUdmSignSuccMessRate',
                                         '5gcUpfShaXinZhuFaSucceRate',
                                         '5gcAmfMaxRegStatusUserNum',
                                         '5gcAmfPageReqNum',
                                         '5gcSmfAvgPduSessNum',
                                         '5gcSmfMaxPduSessNum',
                                         '5gcSmf4GAvgPduSessNum',
                                         '5gcSmf4GMaxPduSessNum',
                                         '5gcSmf45GAvgPduSessNum',
                                         '5gcSmf45GMaxPduSessNum',
                                         '5gcUpfUsrCurrOnlSessNum',
                                         '5gcUpfN3SaveGtp13MessThou',
                                         '5gcUpfN3ReceGtpMaxKbMessRate',
                                         '5gcUpfN3SendGtpMessThou',
                                         '5gcUpfN3SendGtpMaxKbMessRate',
                                         '5gcUpfN6SendGtp13MessThou',
                                         '5gcUpfN6RecUsrPeakMaxKBRate',
                                         '5gcUpfN6SendGtp14MessThou',
                                         '5gcUpfN6SendUsrPeakMaxKBRate',
                                         '5gcUpfN9RecUsrPeakMaxKBRate',
                                         '5gcUpfN9SendUsrPeakMaxKBRate',
                                         '5gcUpf7560Num',
                                         '5gcUdmSubUserNum',
                                         '5gcUdm3gppActiUserNum') and t2.kpiValue < 0.8 * t1.forecastValueLower then 75
                         when t1.kpiId IN ('5gcAmfNotUserReasonAuthSuccRate',
                                         '5gcAmfAttSucRateWithOutUser',
                                         '5gcAmfFinReqSuccRate',
                                         '5gcAmfN2SerReqSccRate',
                                         '5gcAmfSwit5gTo4gSuccRate',
                                         '5gcAmfSwit4gTo5gSuccRate',
                                         '5gcAmfNet4gTo5gReqSuccRate',
                                         '5gcAmfNet5gTo4gSuccRate',
                                         '5gcAmfXnInterSwitSuccRate1',
                                         '5gcAmfN2InterSwitSuccRate1',
                                         '5gcAmfInterSwitInSuccRate1',
                                         '5gcAmfInterSwitOutSuccRate1',
                                         '5gcSmfIMSEstSucRate',
                                         '5gcSmfSessBuiSuccRate',
                                         '5gcSmfFallbackSuccRate',
                                         '5gcUpfPduSessSuccRate',
                                         '5gcUpfXiaXinZhuFaSucceRate',
                                         '5gcUdmAmfSuccMessRate',
                                         '5gcUdmSmfSuccMessRate',
                                         '5gcUdmSignSuccMessRate',
                                         '5gcUpfShaXinZhuFaSucceRate',
                                         '5gcAmfMaxRegStatusUserNum',
                                         '5gcAmfPageReqNum',
                                         '5gcSmfAvgPduSessNum',
                                         '5gcSmfMaxPduSessNum',
                                         '5gcSmf4GAvgPduSessNum',
                                         '5gcSmf4GMaxPduSessNum',
                                         '5gcSmf45GAvgPduSessNum',
                                         '5gcSmf45GMaxPduSessNum',
                                         '5gcUpfUsrCurrOnlSessNum',
                                         '5gcUpfN3SaveGtp13MessThou',
                                         '5gcUpfN3ReceGtpMaxKbMessRate',
                                         '5gcUpfN3SendGtpMessThou',
                                         '5gcUpfN3SendGtpMaxKbMessRate',
                                         '5gcUpfN6SendGtp13MessThou',
                                         '5gcUpfN6RecUsrPeakMaxKBRate',
                                         '5gcUpfN6SendGtp14MessThou',
                                         '5gcUpfN6SendUsrPeakMaxKBRate',
                                         '5gcUpfN9RecUsrPeakMaxKBRate',
                                         '5gcUpfN9SendUsrPeakMaxKBRate',
                                         '5gcUpf7560Num',
                                         '5gcUdmSubUserNum',
                                         '5gcUdm3gppActiUserNum') and t2.kpiValue < 0.9 * t1.forecastValueLower then 55
                         when t1.kpiId IN ('5gcAmfNotUserReasonAuthSuccRate',
                                         '5gcAmfAttSucRateWithOutUser',
                                         '5gcAmfFinReqSuccRate',
                                         '5gcAmfN2SerReqSccRate',
                                         '5gcAmfSwit5gTo4gSuccRate',
                                         '5gcAmfSwit4gTo5gSuccRate',
                                         '5gcAmfNet4gTo5gReqSuccRate',
                                         '5gcAmfNet5gTo4gSuccRate',
                                         '5gcAmfXnInterSwitSuccRate1',
                                         '5gcAmfN2InterSwitSuccRate1',
                                         '5gcAmfInterSwitInSuccRate1',
                                         '5gcAmfInterSwitOutSuccRate1',
                                         '5gcSmfIMSEstSucRate',
                                         '5gcSmfSessBuiSuccRate',
                                         '5gcSmfFallbackSuccRate',
                                         '5gcUpfPduSessSuccRate',
                                         '5gcUpfXiaXinZhuFaSucceRate',
                                         '5gcUdmAmfSuccMessRate',
                                         '5gcUdmSmfSuccMessRate',
                                         '5gcUdmSignSuccMessRate',
                                         '5gcUpfShaXinZhuFaSucceRate',
                                         '5gcAmfMaxRegStatusUserNum',
                                         '5gcAmfPageReqNum',
                                         '5gcSmfAvgPduSessNum',
                                         '5gcSmfMaxPduSessNum',
                                         '5gcSmf4GAvgPduSessNum',
                                         '5gcSmf4GMaxPduSessNum',
                                         '5gcSmf45GAvgPduSessNum',
                                         '5gcSmf45GMaxPduSessNum',
                                         '5gcUpfUsrCurrOnlSessNum',
                                         '5gcUpfN3SaveGtp13MessThou',
                                         '5gcUpfN3ReceGtpMaxKbMessRate',
                                         '5gcUpfN3SendGtpMessThou',
                                         '5gcUpfN3SendGtpMaxKbMessRate',
                                         '5gcUpfN6SendGtp13MessThou',
                                         '5gcUpfN6RecUsrPeakMaxKBRate',
                                         '5gcUpfN6SendGtp14MessThou',
                                         '5gcUpfN6SendUsrPeakMaxKBRate',
                                         '5gcUpfN9RecUsrPeakMaxKBRate',
                                         '5gcUpfN9SendUsrPeakMaxKBRate',
                                         '5gcUpf7560Num',
                                         '5gcUdmSubUserNum',
                                         '5gcUdm3gppActiUserNum') and t2.kpiValue < t1.forecastValueLower then 35
                         when t1.kpiId IN ('5gcSmfAvgSystemLoad',
                                           '5gcAmfAvgSysLoad',
                                           '5gcUpfPduSessModiFailNum',
                                           '5gcUpfSysAvgLoadCpu', 
                                           '5gcUpfSysAvgLoadMe') and t2.kpiValue > 1.4 * t1.forecastValueUpper then 100
                         when t1.kpiId IN ('5gcSmfAvgSystemLoad',
                                           '5gcAmfAvgSysLoad',
                                           '5gcUpfPduSessModiFailNum',
                                           '5gcUpfSysAvgLoadCpu', 
                                           '5gcUpfSysAvgLoadMe') and t2.kpiValue > 1.2 * t1.forecastValueUpper then 75
                         when t1.kpiId IN ('5gcSmfAvgSystemLoad',
                                           '5gcAmfAvgSysLoad',
                                           '5gcUpfPduSessModiFailNum',
                                           '5gcUpfSysAvgLoadCpu', 
                                           '5gcUpfSysAvgLoadMe') and t2.kpiValue > 1.1 * t1.forecastValueUpper then 55
                         when t1.kpiId IN ('5gcSmfAvgSystemLoad',
                                           '5gcAmfAvgSysLoad',
                                           '5gcUpfPduSessModiFailNum',
                                           '5gcUpfSysAvgLoadCpu', 
                                           '5gcUpfSysAvgLoadMe') and t2.kpiValue > t1.forecastValueUpper then 35
                         else 0 end as abnScore
                   ,t2.minId
                   ,t2.vender
                   ,t2.resourceType
            FROM ta_5gc_fcst_futr6h_day_{resource_type}_{day_time} t1
            INNER JOIN t_5gc_perf_vert_5t_{resource_type}_{day_time} t2
            ON  t1.resId = t2.resId AND t1.itemPara = t2.itemPara AND t1.kpiId = t2.kpiId
        )
        SELECT  minId
               ,nodeCode
               ,resId
               ,resourceType
               ,vender
               ,kpiId
               ,kpiValue
               ,forecastValue
               ,forecastValueUpper
               ,forecastValueLower
               ,ifRangeFcst
               ,CASE WHEN round(abnScore,0) >= 0 AND round(abnScore,0) < 40 THEN 1
                     WHEN round(abnScore,0) >= 40 AND round(abnScore,0) < 60 THEN 2
                     WHEN round(abnScore,0) >= 60 AND round(abnScore,0) < 80 THEN 3  ELSE 4 END AS abnLevel
               ,round(abnScore,0)                                                               AS abnScore
               ,substring(current_timestamp(),0,19)                                             AS createTime
        FROM tmp01
    ''')
    logger.debug(result_5gc.show(5))
    ## 指定表名
    result_5gc.createOrReplaceTempView(f"ta_5gc_fcst_result_day_{resource_type}_{day_time}")
    ## 关联操作
    abnormal_5gc = spark.sql(f'''
        WITH tmp01 AS
        (
            SELECT  DISTINCT t1.nodeCode
                   ,t1.resId
                   ,t1.resourceType
                   ,t1.vender
                   ,t1.kpiId
                   ,t1.abnScore*t2.abnScoreWeight AS abnScore
                   ,t1.abnScore as kpi_abnScore
            FROM ta_5gc_fcst_result_day_{resource_type}_{day_time} t1
            LEFT JOIN dim_kpi_feature_5gc_{resource_type}_{day_time} t2
            ON t1.kpiId = t2.kpiId AND t1.resourceType = t2.resourceType
        )
        SELECT  nodeCode
               ,resId
               ,resourceType
               ,vender
               ,cast(round(SUM(abnScore),0) AS int)                                     AS abnScore
               ,CASE WHEN SUM(abnScore) >= 0 AND SUM(abnScore) < 40 THEN 1
                     WHEN SUM(abnScore) >= 40 AND SUM(abnScore) < 60 THEN 2
                     WHEN SUM(abnScore) >= 60 AND SUM(abnScore) < 80 THEN 3  ELSE 4 END AS abnLevel
               ,substring(current_timestamp(),0,19)                                     AS trainingTime
               ,'fcst'                                                                  AS trainingAlgo
               ,CASE WHEN SUM(abnScore) >= 80 THEN round(SUM(case
                     WHEN kpi_abnScore >= 80 THEN 1 ELSE 0 END)*100/COUNT(1),2)  ELSE 0 END  AS abnRate
        FROM tmp01
        GROUP BY  nodeCode
                 ,resId
                 ,resourceType
                 ,vender
            ''')

    logger.debug(abnormal_5gc.show(5))
    # ## 指定表名
    abnormal_5gc.createOrReplaceTempView(f"ta_5gc_abnormal_day_{resource_type}_{day_time}")
    del abnormal_5gc
    window_5gc = spark.sql(f'''
        WITH tmp01 as
        (
            SELECT  nodeCode
                   ,resId
                   ,resourceType
                   ,vender
                   ,cast(abnScore as int) AS abnScore
            FROM ta_5gc_abnormal_day_{resource_type}_{day_time}
            UNION ALL
            SELECT  nodeCode
                   ,resId
                   ,resourceType
                   ,vender
                   ,cast(abnScore as int) AS abnScore
            FROM ta_5gc_abnormal_day_last_{resource_type}_{day_time}
        ), tmp03 as(
        SELECT  nodeCode
               ,resId
               ,resourceType
               ,vender
               ,SUM(abnScore)                                     AS windowScore
               ,CASE WHEN SUM(abnScore) >= 100 THEN 1  ELSE 0 END AS isWindowAlarm
               ,CASE WHEN SUM(abnScore) >= 200 THEN 1  ELSE 0 END AS isWindowAbn
        FROM tmp01
        GROUP BY  nodeCode
                 ,resId
                 ,resourceType
                 ,vender )
        SELECT  '{min_id}' as minId
               ,t2.nodeCode
               ,t2.resId
               ,t2.resourceType
               ,t2.vender
               ,t3.windowScore
               ,t3.isWindowAbn
               ,t3.isWindowAlarm
               ,t2.abnScore
               ,t2.abnLevel
               ,t2.trainingTime
               ,t2.trainingAlgo
               ,t2.abnRate
               ,substring(current_timestamp(),0,19) AS createTime
        FROM ta_5gc_abnormal_day_{resource_type}_{day_time} t2
        LEFT JOIN tmp03 t3
        ON t2.resId = t3.resId AND t2.nodeCode = t3.nodeCode AND t2.vender = t3.vender AND t2.resourceType = t3.resourceType
        ''')
    logger.debug(window_5gc.show(5))
    ## 把结果转成pandas的dataframe输出
    result_5gc_df = result_5gc.toPandas()
    window_5gc_df = window_5gc.toPandas()
    del result_5gc
    del window_5gc

    # 删除临时表
    spark.catalog.dropTempView(f"ta_5gc_fcst_futr6h_day_{resource_type}_{day_time}")
    spark.catalog.dropTempView(f"t_5gc_perf_vert_5t_{resource_type}_{day_time}")
    spark.catalog.dropTempView(f"dim_kpi_feature_5gc_{resource_type}_{day_time}")
    spark.catalog.dropTempView(f"ta_5gc_abnormal_day_last_{resource_type}_{day_time}")
    # spark.catalog.dropTempView(f"dim_5gc_net_info_{resource_type}_{day_time}")
    spark.catalog.dropTempView(f"ta_5gc_fcst_result_day_{resource_type}_{day_time}")
    spark.catalog.dropTempView(f"ta_5gc_abnormal_day_{resource_type}_{day_time}")

    return result_5gc_df, window_5gc_df