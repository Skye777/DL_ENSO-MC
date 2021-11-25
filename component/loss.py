"""
@author: Skye Cui
@file: loss.py
@time: 2021/2/24 14:55
@description: 
"""
import tensorflow as tf
import numpy as np


class Loss(tf.keras.Model):

    def __init__(self, model):
        super(Loss, self).__init__()
        self.model = model

    def call(self, inputs):
        loss = 0
        # for variable in self.model.trainable_variables:
        #     loss += 5e-4 * tf.nn.l2_loss(variable)
        # loss_reg = loss

        ssim = tf.image.ssim(inputs[0], inputs[1], max_val=1)
        loss_ssim = tf.reduce_mean((1.0 - ssim) / 2)
        loss += 9 * loss_ssim

        gdl_loss = self.gdl_loss(tf.transpose(inputs[0], [1, 0, 2, 3, 4]), tf.transpose(inputs[1], [1, 0, 2, 3, 4]))
        loss_gdl = 0.0001 * gdl_loss
        loss += loss_gdl

        l2_loss = tf.reduce_mean(tf.square(inputs[0] - inputs[1]))
        loss_l2 = l2_loss
        loss += 7 * l2_loss

        l1_loss = tf.reduce_mean(tf.abs(inputs[0] - inputs[1]))
        loss_l1 = l1_loss
        loss += l1_loss

        return loss_ssim, loss_gdl, loss_l2, loss_l1, loss

    def gdl_loss(self, gen_frames, gt_frames, alpha=2):
        """
        Calculates the sum of GDL losses between the predicted and ground truth frames.
        @param gen_frames: [seq_len, batch_size, h, w, c]
        @param gt_frames: [seq_len, batch_size, h, w, c]
        @param alpha: The power to which each gradient term is raised.
        @return: The GDL loss.
        """
        print(gen_frames.shape, gt_frames.shape)
        losses = []
        for i in range(len(gen_frames)):
            filter_x = tf.expand_dims(tf.expand_dims(tf.constant([[-1, 1]], dtype=tf.float32), -1), -1)
            filter_y = tf.expand_dims(tf.expand_dims(tf.constant([[1], [-1]], dtype=tf.float32), -1), -1)
            strides = [1, 1]  # stride of (1, 1)
            padding = 'SAME'

            gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
            gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
            gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
            gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

            grad_diff_x = tf.abs(gt_dx - gen_dx)
            grad_diff_y = tf.abs(gt_dy - gen_dy)

            losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))

        # condense into one tensor and avg
        return tf.reduce_mean(tf.stack(losses))


if __name__ == '__main__':
    from model import UConvlstm
    from hparams import Hparams

    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    model = UConvlstm(hp)
    model_loss = Loss(model)
    pred = tf.random.uniform(shape=(hp.batch_size, hp.out_seqlen, hp.height, hp.width, 1))
    target = tf.random.uniform(shape=(hp.batch_size, hp.out_seqlen, hp.height, hp.width, 1))

    loss_ssim, loss_gdl, loss_l2, loss_l1, loss = model_loss([pred, target])
    print(loss_ssim, loss_gdl, loss_l2, loss_l1, loss)

    # gen_dy, gen_dx = tf.image.image_gradients(gen_frames)
    # gt_dy, gt_dx = tf.image.image_gradients(gt_frames)
    # grad_diff_x = tf.abs(gt_dx - gen_dx)
    # grad_diff_y = tf.abs(gt_dy - gen_dy)
