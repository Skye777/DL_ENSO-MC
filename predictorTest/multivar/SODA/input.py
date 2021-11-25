"""
@author: Skye Cui
@file: input.py
@time: 2021/5/11 10:38
@description: 
"""
import tensorflow as tf
import numpy as np

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def get_features():
    features = {
        'input_sst': tf.io.FixedLenFeature([], tf.string),
        'input_t300': tf.io.FixedLenFeature([], tf.string),
        'input_ssh': tf.io.FixedLenFeature([], tf.string),
        'input_taux': tf.io.FixedLenFeature([], tf.string),
        'input_tauy': tf.io.FixedLenFeature([], tf.string),

        'output_sst': tf.io.FixedLenFeature([], tf.string)}
    return features


def parse_fn(example):
    height = hp.height
    width = hp.width
    features = get_features()

    parsed = tf.io.parse_single_example(serialized=example, features=features)
    # print("parsed:", parsed)

    inputs_list = []
    outputs_list = []
    for vrb in hp.input_variables:
        inputs_list.append(tf.reshape(tf.io.decode_raw(parsed['input_'+vrb], tf.float32), [hp.in_seqlen, height, width, 1]))
    for vrb in hp.output_variables:
        outputs_list.append(tf.reshape(tf.io.decode_raw(parsed['output_'+vrb], tf.float32), [hp.out_seqlen, height, width, 1]))
    print("inputs_list:", inputs_list)
    print("outputs_list:", outputs_list)
    # [time, h, w, predictor]
    inputs_list = tf.transpose(tf.squeeze(inputs_list), [1, 2, 3, 0])
    x = inputs_list
    y = outputs_list[0]

    return x, y


def train_input_fn():

    train_filenames = [hp.multivar + f"/SODA/tfRecords/train-in{hp.in_seqlen}_out{hp.out_seqlen}_lead{hp.lead_time}.tfrecords"]
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(parse_fn)
    train_dataset = train_dataset.shuffle(hp.random_seed).batch(hp.batch_size)

    test_filenames = [hp.multivar + f"/SODA/tfRecords/test-in{hp.in_seqlen}_out{hp.out_seqlen}_lead{hp.lead_time}.tfrecords"]
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(parse_fn)
    test_dataset = test_dataset.batch(hp.batch_size)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train, test = train_input_fn()
    print("train:", train)
    print("test:", test)
