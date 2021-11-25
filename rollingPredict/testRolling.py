"""
@author: Skye Cui
@file: testRolling.py
@time: 2021/6/25 10:09
@description: 
"""
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import UConvlstm, StackConvlstm
from component.plot_helper import plot_helper, season_barrier, nino_seq_plot
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
HORIZON = 18


def sst_climatology():
    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst']
    cli_mean_range = range(1, 349)  # (1982-2010)
    cli_mean = np.mean(np.reshape(sst[cli_mean_range, :, :], (-1, 12, hp.height, hp.width)), axis=0)
    return cli_mean  # (12, h, w) from Jan to Dec


def preprocess_helper(h, w, one_sample=False):
    test_month_start = hp.test_month_start
    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst']
    if one_sample:
        scope = range(test_month_start, test_month_start + hp.in_seqlen + hp.lead_time + HORIZON)
    else:
        scope = range(test_month_start, len(sst))
    sst = sst[scope, :, :]
    t300 = np.load(f"{hp.multivar}/GODAS/npz_data/t300.npz")['t300'][scope, :, :]
    uwind = np.load(f"{hp.multivar}/GODAS/npz_data/uwind.npz")['u10'][scope, :, :]
    vwind = np.load(f"{hp.multivar}/GODAS/npz_data/vwind.npz")['v10'][scope, :, :]

    sst_origin = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst'][scope, :, :]

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, h * w))), (-1, h, w))
    uwind = np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, h * w))), (-1, h, w))
    vwind = np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, h * w))), (-1, h, w))
    t300 = np.reshape(scaler.fit_transform(np.reshape(t300, (-1, h * w))), (-1, h, w))

    return sst, t300, uwind, vwind, sst_origin


def nino_seq(ssta):
    # inputs: [sample, time, h, w]
    # outputs: [sample, time]
    nino = []
    for sample in range(len(ssta)):
        n_index = [np.mean(ssta[sample, i, 35:45, 70:120]) for i in range(len(ssta[sample]))]
        nino.append(n_index)
    return np.array(nino)


def load_model():
    sst_model = UConvlstm(hp, num_predictor=4)
    sst_model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_sst_model_file}')
    t300_model = UConvlstm(hp, num_predictor=2)
    t300_model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_t300_model_file}')
    uw_model = UConvlstm(hp, num_predictor=1)
    uw_model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_uw_model_file}')
    vw_model = UConvlstm(hp, num_predictor=2)
    vw_model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_vw_model_file}')
    return sst_model, t300_model, uw_model, vw_model


def load_samples(dic, inp_list, samples_num):
    test_samples = []
    for m in range(samples_num):
        data = []
        for inp in inp_list:
            data.append(dic[inp][m:m + hp.in_seqlen])
        data = np.transpose(data, (1, 2, 3, 0))
        test_samples.append(data)  # (sample, t, h, w, m)
    return test_samples


def predict(horizon=HORIZON):
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
    ssta_true = []
    samples_len = sst.shape[0] - hp.in_seqlen - horizon - hp.lead_time + 1
    for m in range(samples_len):
        pred_start = m + hp.in_seqlen + hp.lead_time
        sst_true.append(sst_origin[pred_start:pred_start + horizon])
        ssta_true.append(ssta_origin[pred_start:pred_start + horizon])

    dic = {'sst': sst, 't300': t300, 'uw': uwind, 'vw': vwind}
    sst_model_samples = load_samples(dic, inp_list=['sst', 't300', 'uw', 'vw'], samples_num=samples_len)
    t300_model_samples = load_samples(dic, inp_list=['t300', 'uw'], samples_num=samples_len)
    uw_model_samples = load_samples(dic, inp_list=['uw'], samples_num=samples_len)
    vw_model_samples = load_samples(dic, inp_list=['sst', 'vw'], samples_num=samples_len)

    sst_preds = []
    ssta_preds = []
    start_pred_mon = 0
    sst_model, t300_model, uw_model, vw_model = load_model()
    print(sst_model)
    print(t300_model)
    print(uw_model)
    print(vw_model)
    for i, sst_model_in in enumerate(sst_model_samples):
        t300_model_in = t300_model_samples[i]
        uw_model_in = uw_model_samples[i]
        vw_model_in = vw_model_samples[i]

        print("test progress: {}/{}".format(i, len(sst_model_samples)))
        sst_pred = []
        ssta_pred = []
        for t in range(horizon):
            sst_out = np.squeeze(sst_model(np.expand_dims(sst_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)
            t300_out = np.squeeze(t300_model(np.expand_dims(t300_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)
            uw_out = np.squeeze(uw_model(np.expand_dims(uw_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)
            vw_out = np.squeeze(vw_model(np.expand_dims(vw_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)

            sst_model_in_new = np.stack((sst_out, t300_out, uw_out, vw_out), axis=-1)  # (h, w, 2)
            sst_model_in = np.concatenate((sst_model_in, np.expand_dims(sst_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]
            t300_model_in_new = np.stack((t300_out, uw_out), axis=-1)  # (h, w, 2)
            t300_model_in = np.concatenate((t300_model_in, np.expand_dims(t300_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]
            uw_model_in_new = np.expand_dims(uw_out, axis=-1)  # (h, w, 1)
            uw_model_in = np.concatenate((uw_model_in, np.expand_dims(uw_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]
            vw_model_in_new = np.stack((sst_out, vw_out), axis=-1)  # (h, w, 2)
            vw_model_in = np.concatenate((vw_model_in, np.expand_dims(vw_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]

            sst = np.squeeze(np.reshape(sst_scaler.inverse_transform(np.reshape(sst_out, (1, h * w))), (1, h, w)))
            sst_pred.append(sst)
            idx = (start_pred_mon + i + t) % 12
            ssta_pred.append(sst - cli_mean[idx])
        sst_preds.append(sst_pred)
        ssta_preds.append(ssta_pred)

    return sst_true, sst_preds, ssta_true, ssta_preds


def predict_one_sample(horizon=12):
    h = hp.height
    w = hp.width
    in_scope = range(0, hp.in_seqlen)
    out_scope = range(hp.in_seqlen + hp.lead_time, hp.in_seqlen + hp.lead_time + horizon)

    sst, t300, uwind, vwind, sst_origin = preprocess_helper(h, w, one_sample=True)

    sst_model_in = np.transpose([sst[in_scope], t300[in_scope], uwind[in_scope], vwind[in_scope]], (1, 2, 3, 0))
    t300_model_in = np.transpose([sst[in_scope], t300[in_scope], uwind[in_scope], vwind[in_scope]], (1, 2, 3, 0))
    uw_model_in = np.transpose([sst[in_scope], t300[in_scope], uwind[in_scope], vwind[in_scope]], (1, 2, 3, 0))
    vw_model_in = np.transpose([sst[in_scope], t300[in_scope], uwind[in_scope], vwind[in_scope]], (1, 2, 3, 0))

    y_true = sst_origin[out_scope, :, :]
    cli_mean = sst_climatology()
    ssta_true = y_true - cli_mean

    sst_scaler = MinMaxScaler()
    sst_scaler.fit_transform(np.reshape(sst_origin, (-1, h * w)))

    sst_pred = []
    ssta_pred = []
    start_pred_mon = 0
    sst_model, t300_model, uw_model, vw_model = load_model()
    for t in range(horizon):
        sst_out = np.squeeze(sst_model(np.expand_dims(sst_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)
        t300_out = np.squeeze(t300_model(np.expand_dims(t300_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)
        uw_out = np.squeeze(uw_model(np.expand_dims(uw_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)
        vw_out = np.squeeze(vw_model(np.expand_dims(vw_model_in, axis=0), training=False))  # (1, 1, h, w, 1)-->(h, w)

        sst_model_in_new = np.stack((sst_out, t300_out, uw_out, vw_out), axis=-1)  # (h, w, 4)
        sst_model_in = np.concatenate((sst_model_in, np.expand_dims(sst_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]
        t300_model_in_new = np.stack((sst_out, t300_out, uw_out, vw_out), axis=-1)  # (h, w, 4)
        t300_model_in = np.concatenate((sst_model_in, np.expand_dims(t300_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]
        uw_model_in_new = np.stack((sst_out, t300_out, uw_out, vw_out), axis=-1)  # (h, w, 4)
        uw_model_in = np.concatenate((sst_model_in, np.expand_dims(uw_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]
        vw_model_in_new = np.stack((sst_out, t300_out, uw_out, vw_out), axis=-1)  # (h, w, 4)
        vw_model_in = np.concatenate((sst_model_in, np.expand_dims(vw_model_in_new, axis=0)), axis=0)[-hp.in_seqlen:]

        sst = np.squeeze(np.reshape(sst_scaler.inverse_transform(np.reshape(sst_out, (1, h * w))), (1, h, w)))
        sst_pred.append(sst)
        idx = (start_pred_mon + t) % 12
        ssta_pred.append(sst - cli_mean[idx])

    return y_true, np.array(sst_pred), ssta_true, np.array(ssta_pred)


def get_rmse(y_true, y_preds):
    # inputs: [sample, time]
    avgrmse = 0
    rmse = []
    for i in range(HORIZON):
        rmse_i = np.sqrt(mean_squared_error(y_pred=y_preds[:, i], y_true=y_true[:, i]))
        rmse.append(rmse_i)
        avgrmse += rmse_i
    avgrmse /= HORIZON
    return avgrmse, rmse


def pcc(y_true, y_preds):
    # inputs: [sample, time]
    avgcor = 0
    cor = []
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_preds, axis=0)

    for i in range(HORIZON):
        fenzi = np.sum((y_true[:, i] - y_true_mean[i]) * (y_preds[:, i] - y_pred_mean[i]))
        fenmu = np.sqrt(
            np.sum((y_true[:, i] - y_true_mean[i]) ** 2) * np.sum((y_preds[:, i] - y_pred_mean[i]) ** 2))
        cor_i = fenzi / fenmu
        cor.append(cor_i)
        avgcor += cor_i
    avgcor /= HORIZON

    return avgcor, cor


def nino_index_curve(results, lead, one_sample=False):
    # inputs: [sample, horizon]
    nino_seq = results[:, lead-1]
    if one_sample:
        nino_seq = nino_seq[:]
    return nino_seq


def metric_nino(nino_true, nino_preds):
    rmse = get_rmse(nino_true, nino_preds)
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
    sst_true, sst_preds, ssta_true, ssta_preds = predict()
    nino_true = nino_seq(np.array(ssta_true))
    nino_preds = nino_seq(np.array(ssta_preds))
    print(nino_true.shape)
    print(nino_preds.shape)

    # metric
    rmse, cor = metric_nino(nino_true, nino_preds)
    print(rmse)
    print(cor)

    # # index seq
    # for i in [1, 3, 6, 9, 12, 18, 24]:
    #     nino_true_seq = nino_index_curve(nino_true, i)
    #     nino_preds_seq = nino_index_curve(nino_preds, i)
    #     print(nino_true_seq.shape)
    #     print(nino_preds_seq.shape)
    #     nino_seq_plot(true=nino_true_seq, pred=nino_preds_seq)

    # season prediction skill
    season_cor = start_season_metric(nino_true, nino_preds, lead=HORIZON)
    # season_barrier(cor, lead=12)
    idx = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    season_barrier(season_cor[:, idx], lead=HORIZON)

    # visualize ssta
    # sst_true, sst_pred, ssta_true, ssta_pred = predict_one_sample()
    # print(ssta_true.min(), ssta_true.max())
    # print(ssta_pred.min(), ssta_pred.max())
    # plot_helper(ssta_true)
    # plot_helper(ssta_pred)

