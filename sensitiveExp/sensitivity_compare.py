"""
@author: Skye Cui
@file: sensitivity_compare.py
@time: 2021/7/10 10:32
@description: 
"""
import numpy as np
import numpy.ma as npm
import os
import csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import UConvlstm, StackConvlstm
from component.plot_helper import plot_helper, nino_seq_plot
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

GROUP_NUM = 30
# warm event (before start time)
START_YEAR = [1986, 1990, 1993, 1996, 2001, 2005, 2008, 2014, 1987, 2006, 2009]
START_MONTH = [1, 6, 7, 4, 4, 5, 6, 6, 6, 7, 6]
# cold event (before start time)
# START_YEAR = [1987, 2006, 2009]
# START_MONTH = [6, 7, 6]

# warm event (before peak time)
# START_YEAR = [1986, 1991, 1993, 1996, 2001, 2005, 2008, 2014]
# START_MONTH = [8, 1, 12, 11, 11, 11, 12, 12]
# cold event (before peak time)
# START_YEAR = [1987, 2007, 2009]
# START_MONTH = [12, 1, 12]

# all regions
AREA_NUM = 6
SENSITIVE_AREA = {"sensitive_area_0": [[35, 55], [70, 95]],
                  "sensitive_area_1": [[37, 52], [35, 55]],
                  "sensitive_area_2": [[65, 80], [60, 80]],
                  "sensitive_area_3": [[65, 80], [100, 115]],
                  "sensitive_area_4": [[0, 10], [35, 50]],
                  "sensitive_area_5": [[0, 10], [72, 92]]}


# nino34
# sensitive_area_1 = [[35, 53], [75, 95]]
# sensitive_area_2 = [[65, 80], [85, 115]]


def get_mask(exp, area):
    sensitive_area = SENSITIVE_AREA["sensitive_area_"+str(area)]
    if exp == "remove":
        mask_value = np.ones([hp.in_seqlen, hp.height, hp.width])
        mask_value[:, sensitive_area[0][0]:sensitive_area[0][1], sensitive_area[1][0]:sensitive_area[1][1]] = 0

    else:
        mask_value = np.zeros([hp.in_seqlen, hp.height, hp.width])
        mask_value[:, sensitive_area[0][0]:sensitive_area[0][1], sensitive_area[1][0]:sensitive_area[1][1]] = 1

    mask = mask_value == 0
    return mask


def sensitive_area_ratio(area):
    sensitive_area = SENSITIVE_AREA["sensitive_area_" + str(area)]
    ratio = ((sensitive_area[0][1] - sensitive_area[0][0]) * (sensitive_area[1][1] - sensitive_area[1][0])) / (hp.height * hp.width)
    # print("ratio:", ratio)
    return ratio


def sst_climatology():
    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst']
    cli_mean_range = range(1, 349)  # (1982-2010)
    cli_mean = np.mean(np.reshape(sst[cli_mean_range, :, :], (-1, 12, hp.height, hp.width)), axis=0)
    return cli_mean  # (12, h, w) from Jan to Dec


def origin_field(event_start_time, h, w):
    scope = range(event_start_time, event_start_time + hp.in_seqlen + hp.out_seqlen + hp.lead_time)

    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst'][scope, :, :]
    t300 = np.load(f"{hp.multivar}/GODAS/npz_data/t300.npz")['t300'][scope, :, :]
    sshg = np.load(f"{hp.multivar}/GODAS/npz_data/sshg.npz")['sshg'][scope, :, :]
    uwind = np.load(f"{hp.multivar}/GODAS/npz_data/uwind.npz")['u10'][scope, :, :]
    vwind = np.load(f"{hp.multivar}/GODAS/npz_data/vwind.npz")['v10'][scope, :, :]

    sst_origin = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst'][scope, :, :]

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, h * w))), (-1, h, w))
    t300 = np.reshape(scaler.fit_transform(np.reshape(t300, (-1, h * w))), (-1, h, w))
    sshg = np.reshape(scaler.fit_transform(np.reshape(sshg, (-1, h * w))), (-1, h, w))
    uwind = np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, h * w))), (-1, h, w))
    vwind = np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, h * w))), (-1, h, w))

    return sst, t300, sshg, uwind, vwind, sst_origin


def add_noise(shape, var):
    if var == "sst":
        variance = 0.1
    else:
        variance = 0.03
    noise = np.random.normal(loc=0.0, scale=variance, size=shape)
    # print(noise.max(), noise.min())
    return noise


def get_errors_field(noise_field, experiment, area):
    errors_field = np.copy(noise_field)
    if experiment == "remove":
        mask = get_mask(experiment, area)
        errors_field = npm.array(errors_field, mask=mask)
        errors_field = errors_field.filled(fill_value=0)

    elif experiment == "retain":
        mask = get_mask(experiment, area)
        errors_field = npm.array(errors_field, mask=mask)
        errors_field = errors_field.filled(fill_value=0)

    return errors_field


def predict(event_initial_time, start_pred_mon, var, area):
    h = hp.height
    w = hp.width
    in_scope = range(0, hp.in_seqlen)
    out_scope = range(hp.in_seqlen + hp.lead_time, hp.in_seqlen + hp.lead_time + hp.out_seqlen)

    sst, t300, sshg, uwind, vwind, sst_origin = origin_field(event_initial_time, h, w)

    # true sst and ssta
    sst_true = sst_origin[out_scope, :, :]
    cli_mean = sst_climatology()
    idx = [i % 12 for i in range(start_pred_mon, start_pred_mon + hp.out_seqlen)]
    ssta_true = sst_true - cli_mean[idx]

    sst_scaler = MinMaxScaler()
    sst_scaler.fit_transform(np.reshape(sst_origin, (-1, h * w)))
    model = UConvlstm(hp, num_predictor=4)
    model.load_weights(f'{hp.delivery_model_dir}/{hp.explain_model_file}')
    noise = add_noise(shape=(hp.in_seqlen, hp.height, hp.width), var=var)

    sst_pred_dic = {}
    ssta_pred_dic = {}
    for experiment in ["all", "remove", "retain"]:
        errors_field = get_errors_field(noise, experiment=experiment, area=area)
        if var == "sst":
            x_in = np.transpose([sst[in_scope] + errors_field, t300[in_scope], uwind[in_scope], vwind[in_scope]], (1, 2, 3, 0))
        else:
            x_in = np.transpose([sst[in_scope], t300[in_scope] + errors_field, uwind[in_scope], vwind[in_scope]], (1, 2, 3, 0))

        y_out = np.squeeze(model(np.expand_dims(x_in, axis=0), training=False))  # (1, t, h, w, 1)-->(t, h, w)
        sst_pred = np.reshape(sst_scaler.inverse_transform(np.reshape(y_out, (hp.out_seqlen, h * w))),
                              (hp.out_seqlen, h, w))
        sst_pred_dic[experiment] = sst_pred
        ssta_pred = sst_pred - cli_mean[idx]
        ssta_pred_dic[experiment] = ssta_pred

    return sst_true, ssta_true, sst_pred_dic, ssta_pred_dic


def nino_seq(ssta):
    # inputs: [time, h, w]
    # outputs: [time]
    nino34 = [np.mean(ssta[i, 35:45, 70:120]) for i in range(len(ssta))]
    return np.array(nino34)


def sst_pred_error(sst_true, sst_pred):
    # inputs: [out_seqlen, h, w]
    seq_errors = []
    for i in range(len(sst_true)):
        seq_errors.append(np.sqrt(mean_squared_error(y_pred=sst_pred[i], y_true=sst_true[i])))
    return np.mean(seq_errors)


def nino_index_pred_error(nino_true, nino_pred):
    # inputs: [out_seqlen]
    return np.mean(np.abs(nino_true - nino_pred))


def evaluate(event_initial_time, start_pred_mon, var, area):
    sst_true, ssta_true, sst_pred_dic, ssta_pred_dic = predict(event_initial_time, start_pred_mon, var, area)

    sst_error = {}
    nino34_error = {}
    for exp in ["all", "remove", "retain"]:
        sst_pred = sst_pred_dic[exp]
        sst_error[exp] = sst_pred_error(sst_true, sst_pred)

        nino34_true = nino_seq(ssta_true)
        nino34_pred = nino_seq(ssta_pred_dic[exp])
        nino34_error[exp] = nino_index_pred_error(nino34_true, nino34_pred)

    rm_reduce_sst_error = (sst_error["all"] - sst_error["remove"]) / sst_error["all"]
    rt_reduce_sst_error = (sst_error["all"] - sst_error["retain"]) / sst_error["all"]
    rm_cost_performance_sst = rm_reduce_sst_error / sensitive_area_ratio(area)
    rt_cost_performance_sst = rt_reduce_sst_error / (1 - sensitive_area_ratio(area))

    rm_reduce_nino_error = (nino34_error["all"] - nino34_error["remove"]) / nino34_error["all"]
    rt_reduce_nino_error = (nino34_error["all"] - nino34_error["retain"]) / nino34_error["all"]
    rm_cost_performance_nino = rm_reduce_nino_error / sensitive_area_ratio(area)
    rt_cost_performance_nino = rt_reduce_nino_error / (1 - sensitive_area_ratio(area))

    return [sst_error["all"], nino34_error["all"], rm_reduce_sst_error, rt_reduce_sst_error, rm_cost_performance_sst,
            rt_cost_performance_sst, rm_reduce_nino_error, rt_reduce_nino_error, rm_cost_performance_nino,
            rt_cost_performance_nino]


if __name__ == '__main__':
    # test
    # for i in range(len(START_YEAR)):
    #     print(START_YEAR[i])
    #     start_time = (START_YEAR[i] - 1982) * 12 + START_MONTH[i]
    #     for var in ["sst", "t300"]:
    #         for area in range(AREA_NUM):
    #             results = evaluate(event_initial_time=start_time, start_pred_mon=START_MONTH[i] - 1, var=var, area=area)
    #             print(results)

    # group experiments for each event and write the results in file
    for i in range(len(START_YEAR)):
        print(START_YEAR[i])
        start_time = (START_YEAR[i] - 1982) * 12 + START_MONTH[i]

        for var in ["sst", "t300"]:
            for area in range(AREA_NUM):
                path = f"results/{START_YEAR[i]+1}_{var}_{area}.csv"
                with open(path, 'w') as f:
                    csv_write = csv.writer(f)
                    csv_head = ["all_sst", "all_nino", "rm_sst", "rt_sst", "rm_cost_sst", "rt_cost_sst", "rm_nino", "rt_nino",
                                "rm_cost_nino", "rt_cost_nino"]
                    csv_write.writerow(csv_head)

                for n in range(GROUP_NUM):
                    results = evaluate(event_initial_time=start_time, start_pred_mon=START_MONTH[i] - 1, var=var, area=area)
                    with open(path, 'a+') as f:
                        csv_write = csv.writer(f)
                        data_row = results
                        csv_write.writerow(data_row)
