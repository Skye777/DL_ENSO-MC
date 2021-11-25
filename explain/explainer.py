"""
@author: Skye Cui
@file: explainer.py
@time: 2021/6/18 18:24
@description: 
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from model import UConvlstm
from component.plot_helper import plot_helper

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.utils.scores import NinoScore, NinoLossScore, CategoricalScore
from tf_explain.core.vanilla_gradients import VanillaGradients

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def sst_climatology():
    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst']
    cli_mean_range = range(1, 349)  # (1982-2010)
    cli_mean = np.mean(np.reshape(sst[cli_mean_range, :, :], (-1, 12, hp.height, hp.width)), axis=0)
    return cli_mean  # (12, h, w) from Jan to Dec


def preprocess_helper(h, w, scope):
    # 198112-201908 (453months)
    sst = np.load(f"{hp.multivar}/GODAS/npz_data/sst.npz")['sst'][scope, :, :]
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


def attention_with_saliency(events):
    h = hp.height
    w = hp.width

    validation_samples = []
    true_values = []
    for start_time in events:
        scope = range(start_time, start_time + hp.in_seqlen + hp.lead_time + hp.out_seqlen)
        sst, t300, uwind, vwind, sst_origin = preprocess_helper(h, w, scope)
        seed_inp = np.transpose([sst[:hp.in_seqlen], t300[:hp.in_seqlen], uwind[:hp.in_seqlen], vwind[:hp.in_seqlen]], (1, 2, 3, 0))
        y_true = tf.expand_dims(sst[hp.in_seqlen + hp.lead_time:hp.in_seqlen + hp.lead_time + hp.out_seqlen].astype(np.float32), axis=-1)  # (t, h, w, 1)
        validation_samples.append(seed_inp)
        true_values.append(y_true)

    model = UConvlstm(hp, hp.num_predictor)
    model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_model_file}')
    # model.layers[-1].activation = tf.keras.activations.linear

    # score = NinoScore(samples=len(events), mon=1)
    score = NinoLossScore(samples=1, y_true=true_values)
    saliency = Saliency(model=model)
    saliency_map = saliency(score=score, seed_input=np.array(validation_samples), smooth_samples=70, smooth_noise=0.30, keepdims=True)
    print(saliency_map.shape)

    # gradcam = Gradcam(model)
    # cam = gradcam(score, seed_input=[validation_sample], penultimate_layer=-1)

    return saliency_map


def model_modifier(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear
    return cloned_model


if __name__ == '__main__':
    # warm event (before start time)
    START_YEAR = [1985, 1990, 1993, 1996, 2001, 2005, 2008, 2013]
    START_MONTH = [8, 5, 9, 4, 5, 8, 6, 10]
    # cold event (before start time)
    # START_YEAR = [1987, 2006, 2009]
    # START_MONTH = [5, 8, 6]

    # warm event (before peak time)
    # START_YEAR = [1986, 1991, 1993, 1996, 2001, 2005, 2008, 2014]
    # START_MONTH = [8, 1, 12, 11, 11, 11, 12, 12]
    # cold event (before peak time)
    # START_YEAR = [1987, 2007, 2009]
    # START_MONTH = [12, 1, 12]

    for i in range(len(START_YEAR)):
        print("progress: {}/{}".format(i+1, len(START_YEAR)))

        start_time = (START_YEAR[i] - 1982) * 12 + START_MONTH[i]
        saliency = attention_with_saliency(events=[start_time])
        saliency_map = saliency[0]

        mon = [str(i % 12 + 1) for i in range(START_MONTH[i], START_MONTH[i] + hp.out_seqlen)]
        titles = [str(START_YEAR[i] + 1) + '.' + m for m in mon]
        plot_helper(saliency_map[:, :, :, 0], titles=titles, year=START_YEAR[i]+1, ids="sst")
        # plot_helper(saliency_map[:, :, :, 1], titles=titles, year=START_YEAR[i]+1, ids="ssh")
        plot_helper(saliency_map[:, :, :, 2], titles=titles, year=START_YEAR[i]+1, ids="t300")
        # plot_helper(saliency_map[:, :, :, 2])
        data = {'sst': np.array(saliency_map[:, :, :, 0]),
                # 'ssh': np.array(saliency_map[:, :, :, 1]),
                't300': np.array(saliency_map[:, :, :, 2])}
        np.savez(f"{hp.saliency_npz}/{START_YEAR[i]+1}.npz", **data)

    # # warm event (peak time)
    # event_1983 = 1   # 198201-198212  strong
    # event_1987 = 4*12 + 8   # 198608-198707  medium
    # event_1992 = 9*12 + 1   # 199101-199112  medium
    # event_1997 = 14*12 + 11   # 199611-199710  strong
    # event_2009 = 27*12       # 200812-200911  medium central
    # event_2015 = 33*12       # 201412-201511  strong
    # event_2018 = 35*12 + 11  # 201711-201810  weak central
    #
    # # cold event (start time)
    # event_1988 = 5*12 + 5    # 198705-198804  strong
    # event_2008 = 24*12 + 8   # 200608-200707  medium
    # event_2010 = 27*12 + 6       # 200906-201005  medium



