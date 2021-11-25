"""
@author: Skye Cui
@file: testDMS_transfer.py
@time: 2021/5/12 20:07
@description: 
"""
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import UConvlstm
from component.plot_helper import plot_helper, season_barrier, nino_seq_plot, metric_plot
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
# HORIZON = 16
# warm event
START_YEAR = [1986, 1990, 1993, 1996, 2001, 2005, 2008, 2014]
START_MONTH = [1, 6, 7, 4, 4, 5, 6, 6]
# cold event
# START_YEAR = [1987, 2006, 2009]
# START_MONTH = [3, 7, 6]


def sst_climatology():
    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst']
    cli_mean_range = range(1, 349)  # (1982-2010)
    cli_mean = np.mean(np.reshape(sst[cli_mean_range, :, :], (-1, 12, hp.height, hp.width)), axis=0)
    return cli_mean  # (12, h, w) from Jan to Dec


def preprocess_helper(h, w, i, one_sample=False):
    # test_month_start = hp.test_month_start
    test_month_start = (START_YEAR[i] - 1982)*12 + START_MONTH[i]
    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst']
    if one_sample:
        scope = range(test_month_start, test_month_start + hp.in_seqlen + hp.out_seqlen + hp.lead_time)
    else:
        scope = range(test_month_start, len(sst))
    sst = sst[scope, :, :]
    t300 = np.load(f"{hp.multivar}/GODAS/npz_data/t300.npz")['t300'][scope, :, :]
    uwind = np.load(f"{hp.multivar}/GODAS/npz_data/uwind.npz")['u10'][scope, :, :]
    vwind = np.load(f"{hp.multivar}/GODAS/npz_data/vwind.npz")['v10'][scope, :, :]

    sst_origin = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst'][scope, :, :]

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, h * w))), (-1, h, w))
    t300 = np.reshape(scaler.fit_transform(np.reshape(t300, (-1, h * w))), (-1, h, w))
    uwind = np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, h * w))), (-1, h, w))
    vwind = np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, h * w))), (-1, h, w))

    return sst, t300, uwind, vwind, sst_origin


def nino_seq(ssta, one_sample=False):
    if one_sample:
        nino = [np.mean(ssta[i, 35:45, 70:120]) for i in range(len(ssta))]

    # inputs: [sample, time, h, w]
    # outputs: [sample, time]
    else:
        nino = []
        for sample in range(len(ssta)):
            n_index = [np.mean(ssta[sample, i, 35:45, 70:120]) for i in range(len(ssta[sample]))]
            nino.append(n_index)

    return np.array(nino)


def predict(model):
    h = hp.height
    w = hp.width
    sst, t300, uwind, vwind, sst_origin = preprocess_helper(h, w)

    cli_mean = sst_climatology()
    ssta_origin = np.reshape(np.reshape(sst_origin[:-8], (-1, 12, h, w)) - cli_mean, (-1, h, w))
    ssta_origin2019 = sst_origin[-8:] - cli_mean[:8]
    ssta_origin = np.concatenate((ssta_origin, ssta_origin2019), axis=0)

    sst_scaler = MinMaxScaler()
    sst_scaler.fit_transform(np.reshape(sst_origin, (-1, h * w)))

    sst_true = []
    sst_preds = []
    ssta_true = []
    ssta_preds = []
    test_samples = []
    start_pred_mon = 0
    for m in range(sst.shape[0] - hp.in_seqlen - hp.out_seqlen + 1):
        data = np.transpose(
            [sst[m:m + hp.in_seqlen], t300[m:m + hp.in_seqlen], uwind[m:m + hp.in_seqlen], vwind[m:m + hp.in_seqlen]],
            (1, 2, 3, 0))
        test_samples.append(data)   # (sample, t, h, w, m)
        pred_start = m + hp.in_seqlen
        sst_true.append(sst_origin[pred_start:pred_start + hp.out_seqlen])
        ssta_true.append(ssta_origin[pred_start:pred_start + hp.out_seqlen])

    # model = UConvlstm(hp)
    # model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_model_file}')
    for x_in in test_samples:
        y_out = np.squeeze(model(np.expand_dims(x_in, axis=0), training=False))  # (1, t, h, w, 1)-->(t, h, w)
        y_pred = np.reshape(sst_scaler.inverse_transform(np.reshape(y_out, (hp.out_seqlen, h * w))),
                            (hp.out_seqlen, h, w))
        sst_preds.append(y_pred)
        idx = [i % 12 for i in range(start_pred_mon, start_pred_mon+hp.out_seqlen)]
        ssta_preds.append(y_pred - cli_mean[idx])
        start_pred_mon += 1

    return sst_true, sst_preds, ssta_true, ssta_preds


def predict_one_sample(i):
    h = hp.height
    w = hp.width
    start_pred_mon = START_MONTH[i] - 1
    in_scope = range(0, hp.in_seqlen)
    out_scope = range(hp.in_seqlen + hp.lead_time, hp.in_seqlen + hp.lead_time + hp.out_seqlen)

    sst, t300, uwind, vwind, sst_origin = preprocess_helper(h, w, i, one_sample=True)
    x_in = np.transpose([sst[in_scope], t300[in_scope], uwind[in_scope], vwind[in_scope]], (1, 2, 3, 0))
    y_true = sst_origin[out_scope, :, :]
    cli_mean = sst_climatology()
    idx = [i % 12 for i in range(start_pred_mon, start_pred_mon + hp.out_seqlen)]
    ssta_true = y_true - cli_mean[idx]

    sst_scaler = MinMaxScaler()
    sst_scaler.fit_transform(np.reshape(sst_origin, (-1, h * w)))

    model = UConvlstm(hp, num_predictor=4)
    model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_model_file}')
    y_out = np.squeeze(model(np.expand_dims(x_in, axis=0), training=False))  # (1, t, h, w, 1)-->(t, h, w)
    sst_pred = np.reshape(sst_scaler.inverse_transform(np.reshape(y_out, (hp.out_seqlen, h * w))),
                          (hp.out_seqlen, h, w))
    ssta_pred = sst_pred - cli_mean[idx]

    return y_true, sst_pred, ssta_true, ssta_pred


def get_rmse(y_true, y_preds):
    # inputs: [sample, time]
    avgrmse = 0
    rmse = []
    for i in range(hp.out_seqlen):
        rmse_i = np.sqrt(mean_squared_error(y_pred=y_preds[:, i], y_true=y_true[:, i]))
        rmse.append(rmse_i)
        avgrmse += rmse_i
    avgrmse /= hp.out_seqlen
    return avgrmse, rmse


def pcc(y_true, y_preds):
    # inputs: [sample, time]
    accskill = 0
    cor = []
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_preds, axis=0)

    for i in range(hp.out_seqlen):
        fenzi = np.sum((y_true[:, i] - y_true_mean[i]) * (y_preds[:, i] - y_pred_mean[i]))
        fenmu = np.sqrt(
            np.sum((y_true[:, i] - y_true_mean[i]) ** 2) * np.sum((y_preds[:, i] - y_pred_mean[i]) ** 2))
        cor_i = fenzi / fenmu
        cor.append(cor_i)
        accskill += cor_i
    accskill /= hp.out_seqlen

    return accskill, cor


def nino_index_curve(results, lead, one_sample=False):
    # inputs: [sample, horizon]
    nino_seq = results[:, lead-1]
    if one_sample:
        nino_seq = nino_seq[:]
    return nino_seq


def metric_nino(nino_true, nino_preds):
    avg, rmse = get_rmse(nino_true, nino_preds)
    skill, cor = pcc(np.array(nino_true), np.array(nino_preds))

    return rmse, cor


def start_season_metric(nino_true, nino_preds, lead):
    # the first predictions start from Jan
    cor = np.zeros((lead, 12))
    for m in range(1, 13, 1):
        for l in range(0, lead, 1):
            true_seq_l_m = []
            pred_seq_l_m = []
            for k in range(m - 1, len(nino_preds), 12):
                true_seq_l_m.append(nino_true[k, l])
                pred_seq_l_m.append(nino_preds[k, l])
            cor[int(l), int(m - 1)] = np.corrcoef(true_seq_l_m, pred_seq_l_m)[0, 1]

    return cor


if __name__ == "__main__":
    path = "D:/Python/transformer/delivery/"
    file = ""
    model = UConvlstm(hp, num_predictor=4)
    model.load_weights(f'{path}/{file}')

    sst_true, sst_preds, ssta_true, ssta_preds = predict(model)
    nino_true = nino_seq(np.array(ssta_true))
    nino_preds = nino_seq(np.array(ssta_preds))
#
    # metric
    rmse, cor = metric_nino(nino_true, nino_preds)
    print(rmse)
    print(cor)
    metric_plot(cor, rmse, title=file)
    # season prediction skill
    season_cor = start_season_metric(nino_true, nino_preds, lead=hp.out_seqlen)
    # season_barrier(cor, lead=12)
    idx = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    season_barrier(season_cor[:, idx], lead=hp.out_seqlen)

    # index seq
    for i in [1, 3, 6, 9, 12]:
        nino_true_seq = nino_index_curve(nino_true, i)
        nino_preds_seq = nino_index_curve(nino_preds, i)
        print(nino_true_seq.shape)
        print(nino_preds_seq.shape)
        nino_seq_plot(true=nino_true_seq, pred=nino_preds_seq)

    # visualize ssta
    for i in range(len(START_YEAR)):
        sst_true, sst_pred, ssta_true, ssta_pred = predict_one_sample(i)
        print(ssta_true.min(), ssta_true.max())
        print(ssta_pred.min(), ssta_pred.max())

        mon = [str(i % 12 + 1) for i in range(START_MONTH[i], START_MONTH[i] + hp.out_seqlen)]
        titles = [str(START_YEAR[i]+1) + '.' + m for m in mon]
        plot_helper(ssta_true, titles=titles, year=START_YEAR[i]+1, ids="true")
        plot_helper(ssta_pred, titles=titles, year=START_YEAR[i]+1, ids="pred")

