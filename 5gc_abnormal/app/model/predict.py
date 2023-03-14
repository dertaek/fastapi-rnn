import json

import elasticsearch.exceptions

from app.db.es_redis_config import ES_SCROLL_ROW_NUM, ES_INDEX_CFG, ES_OUT_INDEX_MAPPING_5GC_FUTURE, BEST
from app.model.TROCH_BASIC_MODEL import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from app.db.elasticsearchwzh import search_with_scroll, create_index, bulk_to_es
import time

logger.add("网元丢失日志.log", filter=lambda x: '[网元丢失]' in x["message"])


def get_data_run_RNN_predict(res_id, *, idx_name: str, type_: str, kpi: list, config_dict: dict):
    query = {
        "track_total_hits": "true",
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "RESID": res_id
                        }
                    },
                    {
                        "terms": {
                            "ITEM_CODE": kpi
                        }
                    }
                ]
            }
        }
    }

    if type_ != "INTERFACE":
        query["query"]["bool"]["must"].append({
            "term": {
                "ITEM_PARA": "-1"
            }
        })
    else:
        query["query"]["bool"]["must_not"] = [{
            "term": {
                "ITEM_PARA": "-1"
            }
        }]

    logger.debug(query)
    params = {
        "_source": "RESID,NODECODE,DEVICE_SET,ITEM_PARA,RECORD_TIME,ITEM_CODE,VALUE",
        "size": ES_SCROLL_ROW_NUM
    }

    filter_path = ["hits.total.value", "hits.hits._source", "_scroll_id"]

    kwargs = {"index": idx_name, "body": query, "params": params, "filter_path": filter_path}
    try:
        rows = search_with_scroll(tag="old", **kwargs)
    except elasticsearch.exceptions.NotFoundError:
        logger.warning(f"[网元丢失]丢失网元id: {res_id}, 当前时间: {time.localtime()}")
        logger.info(f"重新获取网元{res_id}的数据")
        rows = search_with_scroll(tag="old", **kwargs)

    max_time = None
    min_time = None
    data_list = []
    for row in rows:
        data_list.append(row["_source"])
        row_time = time.strptime(row["_source"]["RECORD_TIME"], "%Y-%m-%d %H:%M:%S")
        if max_time is None:
            max_time = row["_source"]["RECORD_TIME"]
            min_time = row["_source"]["RECORD_TIME"]
        else:
            max_time = row["_source"]["RECORD_TIME"] if row_time > time.strptime(max_time,
                                                                                 "%Y-%m-%d %H:%M:%S") else max_time
            min_time = row["_source"]["RECORD_TIME"] if row_time < time.strptime(min_time,
                                                                                 "%Y-%m-%d %H:%M:%S") else min_time

    date_range_df = pd.DataFrame(index=pd.date_range(start=min_time, end=max_time, freq="300s"))
    date_range_df.index = pd.Series([i.to_pydatetime() for i in date_range_df.index])
    logger.debug(data_list)
    df = pd.DataFrame(data_list)
    df = df.drop_duplicates()
    logger.debug(df)
    ITEM_PARA, NODECODE, DEVICE_SET = df["ITEM_PARA"][0], df["NODECODE"][0], df["DEVICE_SET"][0]
    dataset = pd.pivot(df, values='VALUE', index='RECORD_TIME', columns='ITEM_CODE')
    dataset.index = pd.Series([pd.to_datetime(i) for i in dataset.index])
    logger.debug(dataset)
    dataset = date_range_df.join(dataset)
    logger.debug(dataset)
    dataset.dropna(axis=1, how='all', inplace=True)
    dataset = dataset.apply(pd.to_numeric, errors='raise')
    dataset.fillna(dataset.median(), inplace=True)
    dataset.set_index(date_range_df.index, inplace=True)
    dataset.to_csv(f"/app/vol/history_data/{res_id}.csv")
    logger.debug(dataset)
    logger.debug("数据处理完毕")
    if dataset.shape[0] != 0 and dataset.shape[1] != 0:
        # rnn_model = build_keras_RNN(2, 32, dataset.shape[1], True)
        # print(rnn_model)
        # time.sleep(10)
        # while True:
        #     data_list = []
        #     redis_conn = get_redis()
        #     resid_itempara_list = redis_conn.blpop("es:blist", timeout=60)
        #     if not resid_itempara_list:
        #         break
        #     resid_itempara = resid_itempara_list[1]
        #     item_code_set = redis_conn.smembers(resid_itempara)
        #     for item_code in item_code_set:
        #         resid_itempara_code = resid_itempara + ":" + item_code
        #         data_list.append(redis_conn.lrange(resid_itempara_code, 0, -1))
        #     dataset = pd.DataFrame(np.array(data_list).T)
        #     print("开始预测")

        # start_time=datetime.datetime.now()
        # 定义超参
        # Hyper Parameters
        TIME_STEP = 72  # 设置步长
        # ###########################################
        # # 读取数据、数据集划分、数据归一化##
        # ###########################################
        n_feature = dataset.shape[1]  # 特征数量
        reframed = series_to_supervised(dataset, 1, 1)
        values = reframed.values
        # # 训练集、测试集划分
        n_train = int(values.shape[0] * 0.7 / TIME_STEP)
        n_test = int(values.shape[0] * 0.3 / TIME_STEP)
        if n_test > 0:
            train_X, train_y, test_X, test_y, x_scaler, y_scaler = data_processing(values, n_train, n_test, n_feature,
                                                                                   TIME_STEP)

            # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            # ###########################################
            # ##########构建RNN模型#######
            # ###########################################

            class RNN(nn.Module):
                def __init__(self):
                    super(RNN, self).__init__()

                    self.rnn = nn.RNN(
                        input_size=n_feature,
                        hidden_size=config_dict["hidden_size"],
                        dropout=config_dict["dropout"],
                        bias=config_dict["bias"],
                        num_layers=config_dict["num_layers"],
                        batch_first=config_dict["batch_first"],
                        bidirectional=config_dict["bidirectional"],
                        nonlinearity=config_dict["nonlinearity"]
                    )
                    self.lt = nn.ReLU()

                    self.out = nn.Linear(config_dict["hidden_size"], n_feature)

                def forward(self, x, h_state):
                    # x (batch, time_step, input_size)
                    # h_state (n_layers, batch, hidden_size)
                    # r_out (batch, time_step, hidden_size)
                    # x包含很多时间步，比如10个时间步的数据可能一起放进来了，但h_state是最后一个时间步的h_state，r_out包含每一个时间步的output
                    r_out, h_state = self.rnn(x, h_state)
                    r_out = self.lt(r_out)
                    #  r_out.shape: torch.Size([50, 10, 32])
                    #  h_state.shape: torch.Size([1, 50, 32])
                    outs = []  # save all predictions
                    for time_step in range(TIME_STEP):
                        #         for time_step in range(r_out.size(1)):    # calculate output for each time step
                        outs.append(self.out(r_out[:, time_step, :]))
                    print(" outs: {}".format((torch.stack(outs, dim=1)).shape))  # outs: torch.Size([50, 10, 1])
                    return torch.stack(outs, dim=1), h_state

            rnn = RNN()

            optimizer = torch.optim.Adam(rnn.parameters(),
                                         lr=config_dict["learning_rate"])  # optimize all cnn parameters
            loss_func = nn.MSELoss()

            ###########################################
            ##########模型训练#######
            ###########################################
            i = 0
            h_state = None
            l = []  # 损失统计

            for step in range(100):
                i = i + 1
                # 保证scalar类型为Double
                rnn = rnn.double()
                prediction, h_state = rnn(train_X, h_state)  # rnn output
                # 输出是[50, 10, 1]，因为RNN预测每一步都有输出，但理论上来讲应该使用最后一步输出才是最准确的，所以后面会尝试只取最后一步值，而不是直接用了[50, 10, 1]
                # !! next step is important !!
                h_state = h_state.data  # repack the hidden state, break the connection from last iteration
                loss = loss_func(prediction, train_y)  # calculate loss'
                print(loss)
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()
                l.append(loss.item())

            #######绘制损失曲线#########
            # loss_plot(l)

            # 保存模型
            # torch.save(rnn, 'rnn_model.pt')
            # 加载模型
            # rnn = torch.load('rnn4.pt')
            ###########################################
            #######测试集效果验证#########
            ###########################################
            h_state = None
            prediction, h_state = rnn(test_X, h_state)
            loss = loss_func(prediction, test_y)
            print(loss)
            print("预测后数据结构：", prediction.shape, "测试集数据结构：", test_y.shape)
            # 预测数据转换
            s = dataset.columns
            y_test_pre = data_restore(prediction, y_scaler, s)
            y_test_orignal = data_restore(test_y, y_scaler, s)

            # ########绘制测试集拟合曲线#####
            # for j in y_test_pre.columns:
            #     plot_sensor_test(j, y_test_orignal, y_test_pre, N=5000)

            # #########回归误差计算###########
            # yz_test = pd.DataFrame()
            # for j in s:
            #     yz_test = pd.concat([yz_test, reg_metric(y_test_pre[j].values.tolist(), y_test_orignal[j].values.tolist())],
            #                         axis=1,
            #                         ignore_index=True)
            # yz_test = yz_test.T
            # yz_test.columns = ['MAE', 'MSE', 'RMSE', 'R_Square', 'MAPE']
            # print(yz_test)

            ###########################################
            #######向后预测一个步长数据#########
            ###########################################
            test_1 = np.array(test_X)[-TIME_STEP:, :]
            test_1 = torch.tensor(test_1)
            h_state = None
            yhat, h_state = rnn(test_1, h_state)
            # 预测数据还原
            yhat = yhat[0]
            yhat = pd.DataFrame(yhat.detach().numpy())
            yhat = pd.DataFrame(y_scaler.inverse_transform(yhat))  # 反归一化
            yhat.columns = dataset.columns
            print(yhat.shape)

            #########预测数据增加时间########
            time_last = dataset.index.max()  # 获取最后一个时间
            tm_rng = pd.DataFrame(
                pd.date_range(time_last, periods=TIME_STEP + 1, freq='300s').strftime("%Y-%m-%d %H:%M:%S"))
            tm_rng = tm_rng.tail(TIME_STEP).reset_index(drop=True)
            tm_rng.columns = ['RECORD_TIME']
            yhat_predict = pd.concat([tm_rng, yhat], axis=1)
            std = yhat_predict.iloc[:, 1:].std().to_frame()
            std.columns = ["kpiStd"]
            std.index = yhat_predict.iloc[:, 1:].columns
            pd.set_option("display.max_columns", None)
            yhat_predict = pd.melt(yhat_predict, id_vars=['RECORD_TIME'], var_name="kpiId", value_name="forecastValue")
            logger.debug(yhat_predict)
            yhat_predict = pd.merge(yhat_predict, std, left_on="kpiId", right_index=True, how="left")
            df_max_best = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["MAX_BEST"]), :]
            max_temp = df_max_best.assign(
                forecastValueUpper=float(100),
                forecastValueLower=0,
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            max_temp = regular(max_temp)
            logger.debug(max_temp)
            min_temp = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["MIN_BEST"]), :].assign(
                forecastValueUpper=lambda x: 1.1 * (x.forecastValue + 5),
                forecastValueLower=float(0),
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            min_temp = regular(min_temp)
            logger.debug(min_temp)
            scale_incr_temp = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["SCALE_INCR"]), :].assign(
                forecastValueUpper=lambda x: 1.25 * (x.forecastValue + 10),
                forecastValueLower=float(0),
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            scale_incr_temp = positive_regular(scale_incr_temp)
            logger.debug(scale_incr_temp)
            scale_decr_temp = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["SCALE_DECR"]), :].assign(
                forecastValueUpper=lambda x: abs(100 * x.forecastValue) + 10000,
                forecastValueLower=lambda x: 0.75 * x.forecastValue,
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            scale_decr_temp = positive_regular(scale_decr_temp)
            logger.debug(scale_decr_temp)
            yhat_predict = max_temp.append([min_temp, scale_incr_temp, scale_decr_temp])
            yhat_predict = yhat_predict.loc[:,
                           ["resId", "itemPara", "recordTime", "createTime", "kpiId", "forecastValue",
                            "forecastValueUpper", "forecastValueLower", "kpiStd", "nodeCode", "deviceSet",
                            "resourceType"]]
            logger.debug(yhat_predict)
            out_in_es(yhat_predict)
            logger.info("预测已完成")


def regular(df):
    if not df.empty:
        df["forecastValue"] = df.apply(lambda x: 100 if x["forecastValue"] > 100 else x["forecastValue"], axis=1)
        df["forecastValue"] = df.apply(lambda x: 0 if x["forecastValue"] < 0 else x["forecastValue"], axis=1)
        df["forecastValueUpper"] = df.apply(lambda x: 100 if x["forecastValueUpper"] > 100 else x["forecastValueUpper"],
                                            axis=1)
        df["forecastValueUpper"] = df.apply(lambda x: 0 if x["forecastValueUpper"] < 0 else x["forecastValueUpper"],
                                            axis=1)
        df["forecastValueLower"] = df.apply(lambda x: 100 if x["forecastValueLower"] > 100 else x["forecastValueLower"],
                                            axis=1)
        df["forecastValueLower"] = df.apply(lambda x: 0 if x["forecastValueLower"] < 0 else x["forecastValueLower"],
                                            axis=1)
    return df


def positive_regular(df):
    if not df.empty:
        df["forecastValue"] = df.apply(lambda x: 0 if x["forecastValue"] < 0 else x["forecastValue"], axis=1)
        df["forecastValueUpper"] = df.apply(lambda x: 0 if x["forecastValueUpper"] < 0 else x["forecastValueUpper"],
                                            axis=1)
        df["forecastValueLower"] = df.apply(lambda x: 0 if x["forecastValueLower"] < 0 else x["forecastValueLower"],
                                            axis=1)
    return df


def get_raw_idx(df, col, filter_list):
    result = []
    for i in df.index:
        if df.iloc[i, :][col] in filter_list:
            result.append(i)
    return result


def out_in_es(data: pd.DataFrame) -> None:
    datetime_list = [time.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in data["recordTime"]]
    day_list = [dy.tm_mday for dy in datetime_list]
    logger.debug(day_list)
    head_day, tail_day = datetime_list[0], datetime_list[-1]
    logger.debug(head_day)
    logger.debug(tail_day)
    if head_day.tm_mday == tail_day.tm_mday:
        data_json = data.to_json(orient="records")
        data_dict = json.loads(data_json)
        idx_name = ES_INDEX_CFG["output_index_5gc"]["future6h"].replace("{YYYYMMDD}", time.strftime("%Y%m%d", head_day))
        logger.debug(idx_name)
        create_index(idx_name, ES_OUT_INDEX_MAPPING_5GC_FUTURE)
        actions = [{
            "_index": idx_name,
            "_source": record
        } for record in data_dict]
        ret = bulk_to_es(actions)
        if len(data) == ret[0]:
            logger.info(f"数据写入成功")
        else:
            logger.error("数据写入异常")
    else:
        data_head_day = data[pd.Series(day_list) == head_day.tm_mday]
        logger.debug(data_head_day)
        head_data_json = data_head_day.to_json(orient="records")
        head_data_dict = json.loads(head_data_json)
        head_idx_name = ES_INDEX_CFG["output_index_5gc"]["future6h"].replace("{YYYYMMDD}",
                                                                             time.strftime("%Y%m%d", head_day))
        logger.debug(head_idx_name)
        create_index(head_idx_name, ES_OUT_INDEX_MAPPING_5GC_FUTURE)
        actions = [{
            "_index": head_idx_name,
            "_source": record
        } for record in head_data_dict]
        ret = bulk_to_es(actions)
        if len(data_head_day) == ret[0]:
            logger.info(f"前一天数据写入成功")
        else:
            logger.error("前一天数据写入异常")

        data_tail_day = data[pd.Series(day_list) == tail_day.tm_mday]
        logger.debug(data_tail_day)
        tail_data_json = data_tail_day.to_json(orient="records")
        tail_data_dict = json.loads(tail_data_json)
        tail_idx_name = ES_INDEX_CFG["output_index_5gc"]["future6h"].replace("{YYYYMMDD}",
                                                                             time.strftime("%Y%m%d", tail_day))
        logger.debug(tail_idx_name)
        create_index(tail_idx_name, ES_OUT_INDEX_MAPPING_5GC_FUTURE)
        actions = [{
            "_index": tail_idx_name,
            "_source": record
        } for record in tail_data_dict]
        ret = bulk_to_es(actions)
        if len(data_tail_day) == ret[0]:
            logger.info(f"后一天数据写入成功")
        else:
            logger.error("后一天数据写入异常")


def get_rest_data_run_RNN_predict(res_id, *, idx_name: str, type_: str, kpi: list, config_dict: dict):
    origin_data = pd.read_csv(f"/app/vol/history_data/{res_id}.csv", index_col=0)
    last_time = origin_data.index[-1]
    idx_name_new = idx_filter(idx_name, last_time)
    query = {
        "track_total_hits": "true",
        "query": {
            "bool": {
                "filter": {
                    "range": {
                        "RECORD_TIME": {
                            "gt": last_time
                        }
                    }
                },
                "must": [
                    {
                        "term": {
                            "RESID": res_id
                        }
                    },
                    {
                        "terms": {
                            "ITEM_CODE": kpi
                        }
                    }
                ]
            }
        }
    }

    if type_ != "INTERFACE":
        query["query"]["bool"]["must"].append({
            "term": {
                "ITEM_PARA": "-1"
            }
        })
    else:
        query["query"]["bool"]["must_not"] = [{
            "term": {
                "ITEM_PARA": "-1"
            }
        }]

    logger.debug(query)
    params = {
        "_source": "RESID,NODECODE,DEVICE_SET,ITEM_PARA,RECORD_TIME,ITEM_CODE,VALUE",
        "size": ES_SCROLL_ROW_NUM
    }

    filter_path = ["hits.total.value", "hits.hits._source", "_scroll_id"]

    kwargs = {"index": idx_name_new, "body": query, "params": params, "filter_path": filter_path}
    try:
        rows = search_with_scroll(tag="old", **kwargs)
    except elasticsearch.exceptions.NotFoundError:
        logger.warning(f"[网元丢失]丢失网元id: {res_id}, 当前时间: {time.localtime()}")
        logger.info(f"重新获取网元{res_id}的数据")
        rows = search_with_scroll(tag="old", **kwargs)

    max_time = None
    min_time = None
    data_list = []
    for row in rows:
        data_list.append(row["_source"])
        row_time = time.strptime(row["_source"]["RECORD_TIME"], "%Y-%m-%d %H:%M:%S")
        if max_time is None:
            max_time = row["_source"]["RECORD_TIME"]
            min_time = row["_source"]["RECORD_TIME"]
        else:
            max_time = row["_source"]["RECORD_TIME"] if row_time > time.strptime(max_time,
                                                                                 "%Y-%m-%d %H:%M:%S") else max_time
            min_time = row["_source"]["RECORD_TIME"] if row_time < time.strptime(min_time,
                                                                                 "%Y-%m-%d %H:%M:%S") else min_time

    if len(data_list) != 0:
        date_range_df = pd.DataFrame(index=pd.date_range(start=min_time, end=max_time, freq="300s"))
        date_range_df.index = pd.Series([i.to_pydatetime() for i in date_range_df.index])
        logger.debug(data_list)
        df = pd.DataFrame(data_list)
        df = df.drop_duplicates()
        logger.debug(df)
        ITEM_PARA, NODECODE, DEVICE_SET = df["ITEM_PARA"][0], df["NODECODE"][0], df["DEVICE_SET"][0]
        dataset = pd.pivot(df, values='VALUE', index='RECORD_TIME', columns='ITEM_CODE')
        dataset.index = pd.Series([pd.to_datetime(i) for i in dataset.index])
        logger.debug(dataset)
        dataset = date_range_df.join(dataset)
        logger.debug(dataset)
        dataset.dropna(axis=1, how='all', inplace=True)
        dataset = dataset.apply(pd.to_numeric, errors='raise')
        dataset.fillna(dataset.median(), inplace=True)
        dataset.set_index(date_range_df.index, inplace=True)

        origin_data = origin_data.iloc[dataset.shape[0]:, :]
        dataset = pd.concat([origin_data, dataset], axis=0)
        dataset.fillna(dataset.median(), inplace=True)
        dataset.to_csv(f"/app/vol/history_data/{res_id}.csv")

        logger.debug(dataset)
        logger.debug("数据处理完毕")
    else:
        dataset = origin_data

    if dataset.shape[0] != 0 and dataset.shape[1] != 0:
        # rnn_model = build_keras_RNN(2, 32, dataset.shape[1], True)
        # print(rnn_model)
        # time.sleep(10)
        # while True:
        #     data_list = []
        #     redis_conn = get_redis()
        #     resid_itempara_list = redis_conn.blpop("es:blist", timeout=60)
        #     if not resid_itempara_list:
        #         break
        #     resid_itempara = resid_itempara_list[1]
        #     item_code_set = redis_conn.smembers(resid_itempara)
        #     for item_code in item_code_set:
        #         resid_itempara_code = resid_itempara + ":" + item_code
        #         data_list.append(redis_conn.lrange(resid_itempara_code, 0, -1))
        #     dataset = pd.DataFrame(np.array(data_list).T)
        #     print("开始预测")

        # start_time=datetime.datetime.now()
        # 定义超参
        # Hyper Parameters
        TIME_STEP = 72  # 设置步长
        # ###########################################
        # # 读取数据、数据集划分、数据归一化##
        # ###########################################
        n_feature = dataset.shape[1]  # 特征数量
        reframed = series_to_supervised(dataset, 1, 1)
        values = reframed.values
        # # 训练集、测试集划分
        n_train = int(values.shape[0] * 0.7 / TIME_STEP)
        n_test = int(values.shape[0] * 0.3 / TIME_STEP)
        if n_test > 0:
            train_X, train_y, test_X, test_y, x_scaler, y_scaler = data_processing(values, n_train, n_test, n_feature,
                                                                                   TIME_STEP)

            # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            # ###########################################
            # ##########构建RNN模型#######
            # ###########################################

            class RNN(nn.Module):
                def __init__(self):
                    super(RNN, self).__init__()

                    self.rnn = nn.RNN(
                        input_size=n_feature,
                        hidden_size=config_dict["hidden_size"],
                        dropout=config_dict["dropout"],
                        bias=config_dict["bias"],
                        num_layers=config_dict["num_layers"],
                        batch_first=config_dict["batch_first"],
                        bidirectional=config_dict["bidirectional"],
                        nonlinearity=config_dict["nonlinearity"]
                    )
                    self.lt = nn.ReLU()

                    self.out = nn.Linear(config_dict["hidden_size"], n_feature)

                def forward(self, x, h_state):
                    # x (batch, time_step, input_size)
                    # h_state (n_layers, batch, hidden_size)
                    # r_out (batch, time_step, hidden_size)
                    # x包含很多时间步，比如10个时间步的数据可能一起放进来了，但h_state是最后一个时间步的h_state，r_out包含每一个时间步的output
                    r_out, h_state = self.rnn(x, h_state)
                    r_out = self.lt(r_out)
                    #  r_out.shape: torch.Size([50, 10, 32])
                    #  h_state.shape: torch.Size([1, 50, 32])
                    outs = []  # save all predictions
                    for time_step in range(TIME_STEP):
                        #         for time_step in range(r_out.size(1)):    # calculate output for each time step
                        outs.append(self.out(r_out[:, time_step, :]))
                    print(" outs: {}".format((torch.stack(outs, dim=1)).shape))  # outs: torch.Size([50, 10, 1])
                    return torch.stack(outs, dim=1), h_state

            rnn = RNN()

            optimizer = torch.optim.Adam(rnn.parameters(),
                                         lr=config_dict["learning_rate"])  # optimize all cnn parameters
            loss_func = nn.MSELoss()

            ###########################################
            ##########模型训练#######
            ###########################################
            i = 0
            h_state = None
            l = []  # 损失统计

            for step in range(100):
                i = i + 1
                # 保证scalar类型为Double
                rnn = rnn.double()
                prediction, h_state = rnn(train_X, h_state)  # rnn output
                # 输出是[50, 10, 1]，因为RNN预测每一步都有输出，但理论上来讲应该使用最后一步输出才是最准确的，所以后面会尝试只取最后一步值，而不是直接用了[50, 10, 1]
                # !! next step is important !!
                h_state = h_state.data  # repack the hidden state, break the connection from last iteration
                loss = loss_func(prediction, train_y)  # calculate loss'
                print(loss)
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()
                l.append(loss.item())

            #######绘制损失曲线#########
            # loss_plot(l)

            # 保存模型
            # torch.save(rnn, 'rnn_model.pt')
            # 加载模型
            # rnn = torch.load('rnn4.pt')
            ###########################################
            #######测试集效果验证#########
            ###########################################
            h_state = None
            prediction, h_state = rnn(test_X, h_state)
            loss = loss_func(prediction, test_y)
            print(loss)
            print("预测后数据结构：", prediction.shape, "测试集数据结构：", test_y.shape)
            # 预测数据转换
            s = dataset.columns
            y_test_pre = data_restore(prediction, y_scaler, s)
            y_test_orignal = data_restore(test_y, y_scaler, s)

            # ########绘制测试集拟合曲线#####
            # for j in y_test_pre.columns:
            #     plot_sensor_test(j, y_test_orignal, y_test_pre, N=5000)

            # #########回归误差计算###########
            # yz_test = pd.DataFrame()
            # for j in s:
            #     yz_test = pd.concat([yz_test, reg_metric(y_test_pre[j].values.tolist(), y_test_orignal[j].values.tolist())],
            #                         axis=1,
            #                         ignore_index=True)
            # yz_test = yz_test.T
            # yz_test.columns = ['MAE', 'MSE', 'RMSE', 'R_Square', 'MAPE']
            # print(yz_test)

            ###########################################
            #######向后预测一个步长数据#########
            ###########################################
            test_1 = np.array(test_X)[-TIME_STEP:, :]
            test_1 = torch.tensor(test_1)
            h_state = None
            yhat, h_state = rnn(test_1, h_state)
            # 预测数据还原
            yhat = yhat[0]
            yhat = pd.DataFrame(yhat.detach().numpy())
            yhat = pd.DataFrame(y_scaler.inverse_transform(yhat))  # 反归一化
            yhat.columns = dataset.columns
            print(yhat.shape)

            #########预测数据增加时间########
            time_last = dataset.index[-1]  # 获取最后一个时间
            logger.debug(f"time_last:------------------------------{time_last}")
            tm_rng = pd.DataFrame(
                pd.date_range(time_last, periods=TIME_STEP + 1, freq='300s').strftime("%Y-%m-%d %H:%M:%S"))
            tm_rng = tm_rng.tail(TIME_STEP).reset_index(drop=True)
            tm_rng.columns = ['RECORD_TIME']
            yhat_predict = pd.concat([tm_rng, yhat], axis=1)
            std = yhat_predict.iloc[:, 1:].std().to_frame()
            std.columns = ["kpiStd"]
            std.index = yhat_predict.iloc[:, 1:].columns
            pd.set_option("display.max_columns", None)
            yhat_predict = pd.melt(yhat_predict, id_vars=['RECORD_TIME'], var_name="kpiId", value_name="forecastValue")
            logger.debug(yhat_predict)
            yhat_predict = pd.merge(yhat_predict, std, left_on="kpiId", right_index=True, how="left")
            df_max_best = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["MAX_BEST"]), :]
            max_temp = df_max_best.assign(
                forecastValueUpper=float(100),
                forecastValueLower=0,
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            max_temp = regular(max_temp)
            logger.debug(max_temp)
            min_temp = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["MIN_BEST"]), :].assign(
                forecastValueUpper=lambda x: 1.1 * (x.forecastValue + 5),
                forecastValueLower=float(0),
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            min_temp = regular(min_temp)
            logger.debug(min_temp)
            scale_incr_temp = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["SCALE_INCR"]), :].assign(
                forecastValueUpper=lambda x: 1.25 * (x.forecastValue + 10),
                forecastValueLower=float(0),
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            scale_incr_temp = positive_regular(scale_incr_temp)
            logger.debug(scale_incr_temp)
            scale_decr_temp = yhat_predict.iloc[get_raw_idx(yhat_predict, "kpiId", BEST["SCALE_DECR"]), :].assign(
                forecastValueUpper=lambda x: abs(100 * x.forecastValue) + 10000,
                forecastValueLower=lambda x: 0.75 * x.forecastValue,
                createTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                resId=res_id,
                itemPara=ITEM_PARA,
                nodeCode=NODECODE,
                deviceSet=DEVICE_SET,
                recordTime=lambda x: x.RECORD_TIME,
                resourceType=type_)
            scale_decr_temp = positive_regular(scale_decr_temp)
            logger.debug(scale_decr_temp)
            yhat_predict = max_temp.append([min_temp, scale_incr_temp, scale_decr_temp])
            yhat_predict = yhat_predict.loc[:,
                           ["resId", "itemPara", "recordTime", "createTime", "kpiId", "forecastValue",
                            "forecastValueUpper", "forecastValueLower", "kpiStd", "nodeCode", "deviceSet",
                            "resourceType"]]
            logger.debug(yhat_predict)
            out_in_es(yhat_predict)
            logger.info("预测已完成")


def idx_filter(idx_conbin: str, filter_time: str) -> str:
    idx_list = idx_conbin.split(",")
    result_list = [idx for idx in idx_list if
                   time.strptime(idx.split("_")[-1], "%Y%m%d") >= time.strptime(filter_time.split(" ")[0], "%Y-%m-%d")]
    res = ""
    for idx in result_list:
        idx += ","
        res += idx
    return res[0:-1]
