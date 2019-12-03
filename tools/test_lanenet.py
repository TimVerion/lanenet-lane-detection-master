#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]# 三通道的均值


def init_args():
    """
	初始化配置
    """
    parser = argparse.ArgumentParser()
	# 待测试的图片路径
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    # 待测试的模型权重路径
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    # 是否是batch测试
	parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    # batch_size的大小
	parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    # 测试效果存放路径
	parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    # 是否使用gpu
	parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def minmax_scale(input_arr):
    """
    获取矩阵的最大值和最小值并归一化到[0,255]
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, use_gpu):
    """
    :param image_path:一张待测试的图片路径
    :param weights_path:训练好的权重路径
    :param use_gpu:是否使用gpu
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('开始读取图像数据并进行预处理')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)# 使用线性插值
    image = image - VGG_MEAN# 三通道减去均值
    log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)# 初始化为测试
	# 实例化主干网络
    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        # 加载训练好的权重
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: [image]})
        t_cost = time.time() - t_start
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        for i in range(4):
		    # 获取矩阵的最大值和最小值并归一化到[0,255]
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)# 转换成图片的显示格式uint8(1, 256, 512, 4)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])# (256, 512, 3)
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

    sess.close()

    return


def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('开始获取图像文件路径...')
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] 开始图像读取和预处理...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预处理{:d}张图像, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预测{:d}张图像车道线, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
				# 一些传统图像处理方法,对二值化图进行处理
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
				# 将二值化图和实例分割图进行融合,聚类
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name)
                    cv2.imwrite(image_save_path, mask_image)

            log.info('[Epoch:{:d}] 进行{:d}张图像车道线聚类, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # 在单幅图像上测试hnet模型
        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
