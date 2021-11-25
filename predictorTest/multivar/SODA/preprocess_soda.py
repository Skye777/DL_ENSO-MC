"""
@author: Skye Cui
@file: preprocess_sst_dms.py
@time: 2021/5/11 10:17
@description: 
"""
"""
DMS: direct multi-step, don't need to predict all predictors
"""
import os
import json
import random
import numpy as np
import tensorflow as tf
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from progress.bar import PixelBar

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def read_data():
    height = hp.height
    width = hp.width

    sst = np.load(f"{hp.multivar}/SODA/npz_data/sst.npz")['sst']
    t300 = np.load(f"{hp.multivar}/SODA/npz_data/t300.npz")['t300']
    ssh = np.load(f"{hp.multivar}/SODA/npz_data/ssh.npz")['ssh']
    taux = np.load(f"{hp.multivar}/SODA/npz_data/taux.npz")['taux']
    tauy = np.load(f"{hp.multivar}/SODA/npz_data/tauy.npz")['tauy']

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = Normalizer()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, height * width))), (-1, height, width))
    t300 = np.reshape(scaler.fit_transform(np.reshape(t300, (-1, height * width))), (-1, height, width))
    ssh = np.reshape(scaler.fit_transform(np.reshape(ssh, (-1, height * width))), (-1, height, width))
    taux = np.reshape(scaler.fit_transform(np.reshape(taux, (-1, height * width))), (-1, height, width))
    tauy = np.reshape(scaler.fit_transform(np.reshape(tauy, (-1, height * width))), (-1, height, width))

    print(sst.shape)
    print(sst.min(), sst.max())
    print(t300.shape)
    print(t300.min(), t300.max())
    print(ssh.shape)
    print(ssh.min(), ssh.max())
    print(taux.shape)
    print(taux.min(), taux.max())
    print(tauy.shape)
    print(tauy.min(), tauy.max())


# ---------- Helpers ----------
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# ---------- Prepare Data ----------
def parse_npz_and_nc_data():
    height = hp.height
    width = hp.width

    # train_scope = range(0, 114*12)  # 1870.01-1983.12
    sst = np.load(f"{hp.multivar}/SODA/npz_data/sst.npz")['sst']
    t300 = np.load(f"{hp.multivar}/SODA/npz_data/t300.npz")['t300']
    ssh = np.load(f"{hp.multivar}/SODA/npz_data/ssh.npz")['ssh']
    taux = np.load(f"{hp.multivar}/SODA/npz_data/taux.npz")['taux']
    tauy = np.load(f"{hp.multivar}/SODA/npz_data/tauy.npz")['tauy']

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = Normalizer()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, height * width))), (-1, height, width))
    t300 = np.reshape(scaler.fit_transform(np.reshape(t300, (-1, height * width))), (-1, height, width))
    ssh = np.reshape(scaler.fit_transform(np.reshape(ssh, (-1, height * width))), (-1, height, width))
    taux = np.reshape(scaler.fit_transform(np.reshape(taux, (-1, height * width))), (-1, height, width))
    tauy = np.reshape(scaler.fit_transform(np.reshape(tauy, (-1, height * width))), (-1, height, width))

    data = []
    target = []
    for i in range(sst.shape[0] - hp.in_seqlen + 1 - hp.lead_time - hp.out_seqlen - hp.rolling_len):
        data.append({'sst': sst[i:i + hp.in_seqlen + hp.rolling_len].astype(np.float32),
                     't300': t300[i:i + hp.in_seqlen + hp.rolling_len].astype(np.float32),
                     'ssh': ssh[i:i + hp.in_seqlen + hp.rolling_len].astype(np.float32),
                     'taux': taux[i:i + hp.in_seqlen + hp.rolling_len].astype(np.float32),
                     'tauy': tauy[i:i + hp.in_seqlen + hp.rolling_len].astype(np.float32)})

        target_start = i + hp.in_seqlen + hp.lead_time
        target.append({'sst': sst[target_start:target_start + hp.out_seqlen + hp.rolling_len].astype(np.float32)})

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=hp.train_eval_split,
                                                                        random_state=hp.random_seed)
    print(len(train_data), len(test_data), len(train_target), len(test_target))
    return train_data, test_data, train_target, test_target


# ---------- IO ----------
def write_records(data, filename):
    series = data[0]
    target = data[1]
    writer = tf.io.TFRecordWriter(f"{hp.multivar}/SODA/tfRecords/{filename}")

    bar = PixelBar(r'Generating', max=len(data), suffix='%(percent)d%%')
    for s, t in zip(series, target):
        example = tf.train.Example(features=tf.train.Features(feature={
            'input_sst': _bytes_feature(s['sst'].tobytes()),
            'input_t300': _bytes_feature(s['t300'].tobytes()),
            'input_ssh': _bytes_feature(s['ssh'].tobytes()),
            'input_taux': _bytes_feature(s['taux'].tobytes()),
            'input_tauy': _bytes_feature(s['tauy'].tobytes()),

            'output_sst': _bytes_feature(t['sst'].tobytes())
        }))
        writer.write(example.SerializeToString())
        bar.next()
    writer.close()
    bar.finish()


# ---------- Go! ----------
if __name__ == "__main__":
    print("Parsing raw data...")
    train_data, test_data, train_target, test_target = parse_npz_and_nc_data()
    print("Writing TF Records to file...")
    write_records((train_data, train_target), f"train-in{hp.in_seqlen}_out{hp.out_seqlen}_lead{hp.lead_time}.tfrecords")
    write_records((test_data, test_target), f"test-in{hp.in_seqlen}_out{hp.out_seqlen}_lead{hp.lead_time}.tfrecords")

    print("Done!")
    # read_data()
