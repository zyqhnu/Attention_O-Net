#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 20 10:58:04 2020

@author: zyq
"""

import numpy as np
import math
from io1 import loadmarker, savemarker, loadimg, loadmarker
import os
import cv2
import matplotlib.pyplot as plt


def LandmarkGeneratorHeatmap(srcimage, lanmarks, sigma=3.0):
    """
    Generates a numpy array landmark images for the specified points and parameters.
    :param srcimage:src image itk
    :param lanmarks:image landmarks array
    :param sigma:Sigma of Gaussian
    :return:heatmap
    """
    image_size = np.shape(srcimage)
    res_maps = np.zeros(image_size)

    for landmark in lanmarks:
        res_maps = res_maps + onelandmarktoheatmap(srcimage, landmark, sigma)
    return res_maps


def onelandmarktoheatmap(srcimage, coords, sigma, sigma_scale_factor=1.0, size_sigma_factor=5, normalize_center=True):
    """
    Generates a numpy array of the landmark image for the specified point and parameters.
    :param srcimage:input src image
    :param coords:one landmark coords on src image([x], [x, y] or [x, y, z]) of the point.
    :param sigma:Sigma of Gaussian
    :param sigma_scale_factor:Every value of the gaussian is multiplied by this value.
    :param size_sigma_factor:the region size for which values are being calculated
    :param normalize_center:if true, the value on the center is set to scale_factor
                             otherwise, the default gaussian normalization factor is used
    :return:heatmapimage
    """
    # landmark holds the image
    srcimage = np.squeeze(srcimage)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    image_size = np.shape(srcimage)
    assert len(image_size) == len(coords), "image dim is not equal landmark coords dim"
    dim = len(coords)
    heatmap = np.zeros(image_size, dtype=np.float)
    # flip point is form [x, y, z]
    flipped_coords = np.array(coords)
    region_start = (flipped_coords - sigma * size_sigma_factor / 2).astype(int)
    region_end = (flipped_coords + sigma * size_sigma_factor / 2).astype(int)
    # check the region start and region end size is in the image range
    region_start = np.maximum(0, region_start).astype(int)  # 起始不能为0
    region_end = np.minimum(image_size, region_end).astype(int)  # 不能超过图像的大小
    # return zero landmark, if region is invalid, i.e., landmark is outside of image
    if np.any(region_start >= region_end):
        return heatmap
    region_size = (region_end - region_start).astype(int)
    sigma = sigma * sigma_scale_factor
    scale = 1.0
    if not normalize_center:
        scale /= math.pow(math.sqrt(2 * math.pi) * sigma, dim)
    if dim == 1:
        dx = np.meshgrid(range(region_size[0]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        squared_distances = x_diff * x_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0]] = cropped_heatmap[:]
    if dim == 2:
        dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        y_diff = dy + region_start[1] - flipped_coords[1]
        squared_distances = x_diff * x_diff + y_diff * y_diff
        #        pr30int(sigma)
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1]] = cropped_heatmap[:, :]
    if dim == 3:
        dy, dx, dz = np.meshgrid(range(region_size[1]), range(region_size[0]), range(region_size[2]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        y_diff = dy + region_start[1] - flipped_coords[1]
        z_diff = dz + region_start[2] - flipped_coords[2]
        squared_distances = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1],
        region_start[2]:region_end[2]] = cropped_heatmap[:, :, :]
    return heatmap


def array_to_list(markers):
    lanmarkers = []
    # print(markers.shape)
    for i in range(markers.shape[0]):
        a = markers[i, :]  # 二维取行, 和三维不一样.
        #        a=np.expand_dims(a,-1)
        lanmarkers.append(a)
    return lanmarkers


def gen_image_mask(srcimg, seg_image, maps, index, src_img_path, seg_path, maps_path):
    if os.path.exists(src_img_path) is False:
        os.mkdir(src_img_path)
    if os.path.exists(seg_path) is False:
        os.mkdir(seg_path)
    if os.path.exists(maps_path) is False:
        os.mkdir(maps_path)
    filepath = src_img_path + index + ".npy"
    filepath2 = seg_path + index + ".npy"
    filepath3 = maps_path + index + ".npy"
    np.save(filepath, srcimg)
    np.save(filepath2, seg_image)
    np.save(filepath3, maps)


def padding(image, zero_padding=True):
    if zero_padding:
        pimg = np.zeros((image.shape[0] + 8,
                         image.shape[1] + 11))
    else:
        pimg = np.ones((image.shape[0] + 8,
                        image.shape[1] + 11))
        pimg = pimg * 255
    pimg[4:4 + image.shape[0],
    5:5 + image.shape[1]] = image
    return pimg


def depadding(image):
    #    pimg = np.zeros((image.shape[0]-2*margin,
    #                     image.shape[1]-2*margin,
    #                     image.shape[2]-2*margin))
    pimg = image[4:image.shape[0] - 4,
           5:image.shape[1] - 6]
    return pimg


def depadding_new_img(image):
    #    pimg = np.zeros((image.shape[0]-2*margin,
    #                     image.shape[1]-2*margin,
    #                     image.shape[2]-2*margin))
    pimg = image[28:image.shape[0] - 12,
           2:image.shape[1] - 3]
    return pimg


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero, and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    # 计算分位数
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    #  限制最大最小值
    slice = np.clip(slice, t, b)  # 把最大最小值调整到b,t之间

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
    return tmp


def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized)-np.min(imgs_normalized)))
    return imgs_normalized


def guiyihua_fuyi_yi(data):
    a = abs(data)
    _range = np.max(abs(data))
    return data / _range


def guiyihua(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


srcImagePath = './gray_seg_marker_data/training_img/'
segImagePath = './gray_seg_marker_data/trianing_seg_img/'
markerPath = './gray_seg_marker_data/training_marker_acctoseg/' # marker 所在文件夹
srcImageNameList = []

for i in os.listdir(srcImagePath):
    if i.endswith(".tif"):
        srcImageNameList.append(i)
print(len(srcImageNameList))    # 这多写的几行代码 可以用来确认数量是否正确

for j in range(len(srcImageNameList)):
    imgName = srcImageNameList[j]
    img = loadimg(srcImagePath + imgName)
    img_seg = loadimg(segImagePath + imgName)    # 分割图和原图同名,所以也是img1.

    imgarray = img[:, :, 0]
    imgarray_seg = img_seg[:, :, 0]

    markers1 = loadmarker(markerPath + imgName + '.marker')     # marker文件名称为 xxx.marker.
    markers1 = markers1[:, :2] - 1     # marker 文件里是从1开始的,而数组是从0开始的,所以marker比array多了1.
    markers1 = array_to_list(markers1)
    maps1 = LandmarkGeneratorHeatmap(imgarray, markers1, sigma=1.5)

    maps1=depadding_new_img(maps1)  # 可以不删除边缘，但要保证所有的图像大小是一样的

    imgarray=depadding_new_img(imgarray)
    img_res1 = dataset_normalized(imgarray)

    img_seg_res1 = imgarray_seg/255
    img_seg_res1 = depadding_new_img(img_seg_res1)

    print(img_res1.max(), img_res1.min())
    # plt.figure(1)
    # plt.imshow(maps1+img_res1)
    # plt.figure(2)
    # plt.imshow(maps1 + img_seg_res1)
    # plt.figure(3)
    # plt.imshow(img_res1 + img_seg_res1)
    # plt.show()

    img_res2 = cv2.flip(img_res1, 1)    # 翻转
    img_res3 = cv2.flip(img_res1, 0)
    img_res4 = cv2.flip(img_res1, -1)

    img_seg_res2 = cv2.flip(img_seg_res1, 1)
    img_seg_res3 = cv2.flip(img_seg_res1, 0)
    img_seg_res4 = cv2.flip(img_seg_res1, -1)


    maps2 = cv2.flip(maps1, 1)
    maps3 = cv2.flip(maps1, 0)
    maps4 = cv2.flip(maps1, -1)

    print(img_res1.shape, img_res2.shape, img_res3.shape,img_res4.shape,maps1.shape,maps2.shape,maps3.shape,maps4.shape)

    a = './train_data/img/'
    b = './train_data/img_seg/'
    c = './train_data/label/'

    imgName = imgName[:-4]
    gen_image_mask(img_res1, img_seg_res1, maps1, imgName + '1', a, b,c)
    gen_image_mask(img_res2, img_seg_res2, maps2, imgName + '2', a, b,c)
    gen_image_mask(img_res3, img_seg_res3, maps3, imgName + '3', a, b,c)
    gen_image_mask(img_res4, img_seg_res4, maps4, imgName + '4', a, b,c)

    # gen_image_mask(img_res2, maps2, imgName + '2', a, b)
    # gen_image_mask(img_res3, maps3, imgName + '3', a, b)
    # gen_image_mask(img_res4, maps4, imgName + '4', a, b)