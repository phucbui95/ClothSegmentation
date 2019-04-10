from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import cv2
from PIL import Image


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *

N_CLASSES = 20


# ------------------------network setting---------------------
#################################################

##  refine net version 4.   07.17

def pose_net(image, name):
    with tf.variable_scope(name) as scope:
        is_BN = False
        pose_conv1 = conv2d(image, 512, 3, 1, relu=True, bn=is_BN,
                            name='pose_conv1')
        pose_conv2 = conv2d(pose_conv1, 512, 3, 1, relu=True, bn=is_BN,
                            name='pose_conv2')
        pose_conv3 = conv2d(pose_conv2, 256, 3, 1, relu=True, bn=is_BN,
                            name='pose_conv3')
        pose_conv4 = conv2d(pose_conv3, 256, 3, 1, relu=True, bn=is_BN,
                            name='pose_conv4')
        pose_conv5 = conv2d(pose_conv4, 256, 3, 1, relu=True, bn=is_BN,
                            name='pose_conv5')
        pose_conv6 = conv2d(pose_conv5, 256, 3, 1, relu=True, bn=is_BN,
                            name='pose_conv6')

        pose_conv7 = conv2d(pose_conv6, 512, 1, 1, relu=True, bn=is_BN,
                            name='pose_conv7')
        pose_conv8 = conv2d(pose_conv7, 16, 1, 1, relu=False, bn=is_BN,
                            name='pose_conv8')

        return pose_conv8, pose_conv6


def pose_refine(pose, parsing, pose_fea, name):
    with tf.variable_scope(name) as scope:
        is_BN = False
        # 1*1 convolution remaps the heatmaps to match the number of channels of the intermediate features.
        pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
        parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN,
                         name='parsing_remap')
        # concat
        pos_par = tf.concat([pose, parsing, pose_fea], 3)
        conv1 = conv2d(pos_par, 512, 3, 1, relu=True, bn=is_BN, name='conv1')
        conv2 = conv2d(conv1, 256, 5, 1, relu=True, bn=is_BN, name='conv2')
        conv3 = conv2d(conv2, 256, 7, 1, relu=True, bn=is_BN, name='conv3')
        conv4 = conv2d(conv3, 256, 9, 1, relu=True, bn=is_BN, name='conv4')

        conv5 = conv2d(conv4, 256, 1, 1, relu=True, bn=is_BN, name='conv5')
        conv6 = conv2d(conv5, 16, 1, 1, relu=False, bn=is_BN, name='conv6')

        return conv6, conv4


def parsing_refine(parsing, pose, parsing_fea, name):
    with tf.variable_scope(name) as scope:
        is_BN = False
        pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
        parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN,
                         name='parsing_remap')

        par_pos = tf.concat([parsing, pose, parsing_fea], 3)
        parsing_conv1 = conv2d(par_pos, 512, 3, 1, relu=True, bn=is_BN,
                               name='parsing_conv1')
        parsing_conv2 = conv2d(parsing_conv1, 256, 5, 1, relu=True, bn=is_BN,
                               name='parsing_conv2')
        parsing_conv3 = conv2d(parsing_conv2, 256, 7, 1, relu=True, bn=is_BN,
                               name='parsing_conv3')
        parsing_conv4 = conv2d(parsing_conv3, 256, 9, 1, relu=True, bn=is_BN,
                               name='parsing_conv4')

        parsing_conv5 = conv2d(parsing_conv4, 256, 1, 1, relu=True, bn=is_BN,
                               name='parsing_conv5')
        parsing_human1 = atrous_conv2d(parsing_conv5, 20, 3, rate=6,
                                       relu=False, name='parsing_human1')
        parsing_human2 = atrous_conv2d(parsing_conv5, 20, 3, rate=12,
                                       relu=False, name='parsing_human2')
        parsing_human3 = atrous_conv2d(parsing_conv5, 20, 3, rate=18,
                                       relu=False, name='parsing_human3')
        parsing_human4 = atrous_conv2d(parsing_conv5, 20, 3, rate=24,
                                       relu=False, name='parsing_human4')
        parsing_human = tf.add_n(
            [parsing_human1, parsing_human2, parsing_human3, parsing_human4],
            name='parsing_human')

        return parsing_human, parsing_conv4


#################################################

class LIP_SSL_Model:
    def __init__(self, fine_width, fine_height):
        self.fine_width = fine_width
        self.fine_height = fine_height

    def forward(self, image_batch_origin):
        return self.build_model_(image_batch_origin)

    def build_model_(self, image_batch_origin):
        """ Build model """
        # image, image_rev = inputs
        w, h = self.fine_width, self.fine_height

        # image_batch_origin = tf.stack([image, image_rev])
        image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
        image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
        image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])
        
        # Create network.
        with tf.variable_scope('', reuse=False):
            net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
        with tf.variable_scope('', reuse=True):
            net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
        with tf.variable_scope('', reuse=True):
            net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)

        
        # parsing net
        parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
        parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
        parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

        parsing_out1_100 = net_100.layers['fc1_human']
        parsing_out1_075 = net_075.layers['fc1_human']
        parsing_out1_125 = net_125.layers['fc1_human']

        # pose net
        resnet_fea_100 = net_100.layers['res4b22_relu']
        resnet_fea_075 = net_075.layers['res4b22_relu']
        resnet_fea_125 = net_125.layers['res4b22_relu']

        with tf.variable_scope('', reuse=False):
            pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
            pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
            parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
            parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')

        with tf.variable_scope('', reuse=True):
            pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
            pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
            parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
            parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')

        with tf.variable_scope('', reuse=True):
            pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
            pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
            parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
            parsing_out3_125, parsing_fea3_125 = parsing_refine(parsing_out2_125, pose_out2_125, parsing_fea2_125, name='fc3_parsing')


        parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3,]),
                                               tf.image.resize_images(parsing_out1_075, tf.shape(image_batch_origin)[1:3,]),
                                               tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
        parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3,]),
                                               tf.image.resize_images(parsing_out2_075, tf.shape(image_batch_origin)[1:3,]),
                                               tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
        parsing_out3 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3,]),
                                               tf.image.resize_images(parsing_out3_075, tf.shape(image_batch_origin)[1:3,]),
                                               tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)

        raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
        head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
        tail_list = tf.unstack(tail_output, num=20, axis=2)
        tail_list_rev = [None] * 20
        for xx in range(14):
            tail_list_rev[xx] = tail_list[xx]
        tail_list_rev[14] = tail_list[15]
        tail_list_rev[15] = tail_list[14]
        tail_list_rev[16] = tail_list[17]
        tail_list_rev[17] = tail_list[16]
        tail_list_rev[18] = tail_list[19]
        tail_list_rev[19] = tail_list[18]
        tail_output_rev = tf.stack(tail_list_rev, axis=2)
        tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

        raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
        raw_output_all = tf.expand_dims(raw_output_all, dim=0)
        raw_output_all = tf.argmax(raw_output_all, dimension=3)
        pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.
        return pred_all

    def load_weights(self, restore_var, sess, path):
        loader = tf.train.Saver(var_list=restore_var)
        if load(loader, sess, path):
            print("Loaded model successfully")
        else:
            print("Failed to load model from checkpoint")