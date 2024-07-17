#SWATHI-SHREE-NARASHIMAN/Lensless-3D-imaging-EPFL-/Samah-modified
'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import tensorflow as tf
from tf_util import *
import numpy as np


def kl_divergence_loss(p,q):
    p=tf.nn.softmax(p)
    q=tf.nn.softmax(q)
    kl_div = tf.reduce_sum(p * tf.log(p / q), axis=1)
    return tf.reduce_mean(kl_div)



    
    

class Model:
    def __init__(self, inputs, npts, gt, alpha, beta, gt_class):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse

        #Getting the two encoder representations
        self.features, self.sub_inputs, self.sub_features = self.create_encoder(inputs, npts)
        self.gt_features, self.gt_sub_inputs, self.gt_sub_features=self.create_encoder(gt,npts)

        #Since it is not VAE, KL Divergence can't be applied - applying MSE loss
        self.correlation_loss=self.correlation_mse_loss(features, gt_features)
        
        self.logits = self.creat_classifier(self.features, 3)
        self.coarse, self.fine = self.create_decoder(self.features, self.sub_inputs, self.sub_features)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha, beta, gt_class)
        #Presenting the net loss term by keeping alpha and beta (proportions of the two losses) 1
        self.total_loss=self.loss+self.correlation_loss
        self.outputs = self.fine
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder(self, inputs, npts):
        #subsample input points randomly to have 1024 point 
        sub_inputs = inputs
        #reshape sub_inputs to have dims [-1, npts, 3]
        sub_inputs = tf.reshape(sub_inputs, [npts.shape[0], 3000, 3])
        sub_inputs = tf.gather(sub_inputs, tf.random_shuffle(tf.range(tf.shape(sub_inputs)[1]))[:self.num_coarse], axis=1)
        npts_ = np.array([self.num_coarse])
        npts_ = tf.tile(npts_, [npts.shape[0]//1])

        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = point_maxpool(features, npts, keepdims=True)
            features_global = point_unpool(features_global, npts)
            features = tf.concat([features, features_global], axis=2)

        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            #sub_features = features
            features = point_maxpool(features, npts)

        with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE):
            sub_ = tf.reshape(sub_inputs, [1, -1, 3])
            features_ = mlp_conv(sub_, [128, 256])
            features_global_ = point_maxpool(features_, npts_, keepdims=True)
            features_global_ = point_unpool(features_global_, npts_)
            features_ = tf.concat([features_, features_global_], axis=2)
        with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE):
            sub_features = mlp_conv(features_, [512, 1024])
        return features, sub_inputs, sub_features

    def create_decoder(self, features, sub_inputs, sub_features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])
            #append sub_inputs to coarse 
            #coarse = tf.concat([coarse, sub_inputs], axis=1)

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])
            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])
            sub_features = tf.reshape(sub_features, [features.shape[0], -1, self.num_coarse])
            sub_features = tf.tile(sub_features, [1, self.num_fine//sub_features.shape[1], 1])
            feat = tf.concat([grid_feat, point_feat, global_feat, sub_features], axis=2)
            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            fine = mlp_conv(feat, [512, 512, 3]) + center
        return coarse, fine

    def create_loss(self, coarse, fine, gt, alpha, beta, gt_class):
        gt_ds = gt[:, :coarse.shape[1], :]
        loss_coarse = earth_mover(coarse, gt_ds)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        
        loss_class = tf.losses.sparse_softmax_cross_entropy(labels=gt_class, logits=self.logits)
        loss = loss_coarse + alpha * loss_fine + beta*loss_class
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]

    def creat_classifier(self, features, num_classes):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            logits = mlp(features, [512, 256, num_classes])
        return logits
    
    def class_loss (self, gt_class):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_class, logits=self.logits)
        return loss

    def correlation_mse_loss(self, features1, features2):
        return return tf.reduce_mean(tf.square(features1 - features2))

