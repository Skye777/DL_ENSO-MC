"""
@author: Skye Cui
@file: layers.py
@time: 2021/2/22 13:43
@description: 
"""
import tensorflow as tf


class ConvAttention(tf.keras.layers.Layer):
    def __init__(self, l, h, w, c, k):
        super(ConvAttention, self).__init__()
        self.reshape = tf.keras.layers.Reshape((l, w * h * c))
        self.layer1 = tf.keras.layers.Dense(units=k, activation='tanh')
        self.layer2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None):
        outputs = self.layer2(self.layer1(self.reshape(inputs)))
        outputs = tf.nn.softmax(outputs, axis=-2)
        return outputs


class WeightedSumBlock(tf.keras.layers.Layer):
    def __init__(self, l, h, w, c):
        super(WeightedSumBlock, self).__init__()
        self.l = l
        self.add = tf.keras.layers.Add()
        self.reshape1 = tf.keras.layers.Reshape((l, w * h * c))
        self.reshape2 = tf.keras.layers.Reshape((h, w, c))

    def call(self, inputs, training=None):
        inputs, alpha = inputs
        inputs = self.reshape1(inputs)
        info = tf.multiply(alpha, inputs)
        info = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=self.l, axis=-2))(info)
        outputs = tf.keras.layers.add(info)
        outputs = self.reshape2(outputs)
        return outputs


class ConvlstmMaxPoolBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool_size, strides, t, h, w, c):
        super(ConvlstmMaxPoolBlock, self).__init__()
        self.convlstm = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                   return_sequences=True, return_state=True,
                                                   activation=tf.keras.layers.LeakyReLU())
        self.max_pool = tf.keras.layers.MaxPool3D(pool_size=(1, pool_size, pool_size), strides=(1, strides, strides))
        self.bn = tf.keras.layers.BatchNormalization()
        self.alpha = ConvAttention(t, h, w, c, k=16)
        self.get_feature_maps = WeightedSumBlock(t, h, w, c)

    def call(self, inputs, skip_layer=None):
        out, state_h, state_c = self.convlstm(inputs)
        bn_out = self.bn(out)
        alpha = self.alpha(bn_out)
        skip_layer_feature_map = self.get_feature_maps([bn_out, alpha])
        pool_out = self.max_pool(bn_out)
        return pool_out, (state_h, state_c), skip_layer_feature_map


class ConvlstmTransBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, up_size, strategy):
        super(ConvlstmTransBlock, self).__init__()
        self.strategy = strategy

        self.deconvlstm = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                     return_sequences=True, activation=tf.keras.layers.LeakyReLU())
        self.up_sampling3d = tf.keras.layers.UpSampling3D(size=(1, up_size, up_size))
        self.bn = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        inputs, hidden_state, map = inputs
        T = tf.shape(inputs)[1]
        skip_layer = tf.tile(tf.expand_dims(map, 1), [1, T, 1, 1, 1])

        up_out = self.up_sampling3d(inputs)
        deconv_out = self.deconvlstm(up_out, initial_state=hidden_state)
        bn_out = self.bn(deconv_out)
        out = tf.keras.layers.Concatenate()([skip_layer, bn_out])
        return out

