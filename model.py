"""
@author: Skye Cui
@file: model.py
@time: 2021/2/23 15:55
@description: 
"""
import tensorflow as tf
from component.layers import ConvlstmMaxPoolBlock, ConvlstmTransBlock, ConvAttention, WeightedSumBlock


def UConvlstm(hp, num_predictor):
    filter_params = [8, 16, 32, 64]

    inp = tf.keras.Input(shape=(hp.in_seqlen, hp.height, hp.width, num_predictor))
    out, state_stage1, map1 = ConvlstmMaxPoolBlock(filters=filter_params[0], kernel_size=3, pool_size=2, strides=2,
                                                   t=hp.in_seqlen, h=80, w=160, c=filter_params[0])(inp)  # map1 (b, 80, 160, 8) out(b, t, 40, 80, 4)
    out, state_stage2, map2 = ConvlstmMaxPoolBlock(filters=filter_params[1], kernel_size=3, pool_size=2, strides=2,
                                                   t=hp.in_seqlen, h=40, w=80, c=filter_params[1])(out)  # map2 (b, 40, 80, 16) out(b, t, 20, 40, 8)
    out, state_stage3, map3 = ConvlstmMaxPoolBlock(filters=filter_params[2], kernel_size=3, pool_size=2, strides=2,
                                                   t=hp.in_seqlen, h=20, w=40, c=filter_params[2])(out)  # map2 (b, 20, 40, 32) out(b, t, 10, 20, 16)

    alpha = ConvAttention(l=hp.in_seqlen, h=10, w=20, c=filter_params[2], k=16)(out)
    out = WeightedSumBlock(l=hp.in_seqlen, h=10, w=20, c=filter_params[2])([out, alpha])

    mid_state = tf.keras.layers.Conv2D(filters=filter_params[3], kernel_size=5, strides=5,
                                       activation=tf.keras.layers.LeakyReLU())(out)  # (b, 2, 4, 32)

    # (b, 2, 4, 32) --> (b, 2, 4, 32)
    deconv_out = tf.keras.layers.Conv2DTranspose(filters=filter_params[-1], kernel_size=5, strides=5,
                                                 activation=tf.keras.layers.LeakyReLU())(mid_state)
    deconv_out = tf.keras.layers.BatchNormalization()(deconv_out)
    deconv_out = tf.tile(tf.expand_dims(deconv_out, 1), [1, hp.out_seqlen, 1, 1, 1])
    out = ConvlstmTransBlock(filters=filter_params[-2], kernel_size=3, up_size=2, strategy=hp.strategy)([deconv_out, state_stage3, map3])
    out = ConvlstmTransBlock(filters=filter_params[-3], kernel_size=3, up_size=2, strategy=hp.strategy)([out, state_stage2, map2])
    out = ConvlstmTransBlock(filters=filter_params[-4], kernel_size=3, up_size=2, strategy=hp.strategy)([out, state_stage1, map1])
    out = tf.keras.layers.ConvLSTM2D(filters=hp.num_output, kernel_size=3, padding='same', return_sequences=True, name="pred")(out)
    model = tf.keras.Model(inp, out)
    return model


if __name__ == '__main__':
    from hparams import Hparams

    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    # inputs = tf.random.uniform(shape=(hp.batch_size, hp.in_seqlen, hp.height, hp.width, hp.num_predictor))
    model = UConvlstm(hp, 4)
    print(model.summary())

