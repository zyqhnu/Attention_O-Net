#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:55:42 2019

@author: zyq
"""

import scipy.ndimage as ndimg
import numpy as np
from numba import jit
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import os


def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

@jit  # trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    return rst

# @jit  # fill a node (may be two or more points)

def fill(img, msk, p, nbs, buf):
    msk[p] = 3
    buf[0] = p
    back = img[p]
    cur = 0
    s = 1
    while cur < s:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == back and msk[cp] == 1:
                msk[cp] = 3
                buf[s] = cp
                s += 1
                if s == len(buf):
                    buf[:s - cur] = buf[cur:]
                    s -= cur
                    cur = 0
        cur += 1
    # msk[p] = 3

# @jit  # my mark

def mark(img, msk, buf, mode):  # mark the array use (0, 1, 2)
    omark = msk
    nbs = neighbors(img.shape)
    idx = np.zeros(1024 * 128, dtype=np.int64)
    img = img.ravel()  # 降维
    msk = msk.ravel()  # 降维
    s = 0
    for p in range(len(img)):
        if msk[p] != 1: continue
        flag = False
        for dp in nbs:
            if mode and img[p + dp] > img[p]:
                flag = True
                break
            elif not mode and img[p + dp] < img[p]:
                flag = True
                break

        if flag:
            continue
        else:
            fill(img, msk, p, nbs, buf)
        idx[s] = p
        s += 1
        if s == len(idx): break
    # plt.imshow(omark, cmap='gray')
    return idx[:s].copy()

def filter(img, msk, idx, bur, tor, mode):
    omark = msk
    nbs = neighbors(img.shape)
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    msk = msk.ravel()

    arg = np.argsort(img[idx])[::-1 if mode else 1]

    for i in arg:
        if msk[idx[i]] != 3:
            idx[i] = 0
            continue
        cur = 0
        s = 1
        bur[0] = idx[i]
        while cur < s:
            p = bur[cur]
            if msk[p] == 2:
                idx[i] = 0
                break

            for dp in nbs:
                cp = p + dp
                if msk[cp] == 0 or cp == idx[i] or msk[cp] == 4: continue
                if mode and img[cp] < img[idx[i]] - tor: continue
                if not mode and img[cp] > img[idx[i]] + tor: continue
                bur[s] = cp
                s += 1
                if s == 1024 * 128:
                    cut = cur // 2
                    msk[bur[:cut]] = 2
                    bur[:s - cut] = bur[cut:]
                    cur -= cut
                    s -= cut

                if msk[cp] != 2: msk[cp] = 4
            cur += 1
        msk[bur[:s]] = 2
        # plt.imshow(omark, cmap='gray')

    return idx2rc(idx[idx > 0], acc)

def find_maximum(img, tor, mode=True):
    msk = np.zeros_like(img, dtype=np.uint8)
    msk[tuple([slice(1, -1)] * img.ndim)] = 1  # 图片最外一层取0

    # plt.imshow(msk, cmap='gray')

    buf = np.zeros(1024 * 128, dtype=np.int64)
    omark = msk
    idx = mark(img, msk, buf, mode)
    # plt.imshow(msk, cmap='gray')
    idx = filter(img, msk, idx, buf, tor, mode)
    return idx

def meanshif(X):
    #    X = loadmarkera('/media/ttt/Elements/TanYinghui/lung1/10_rb5_2d_3.marker')
    marker_r = []
    # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=1067)
    bandwidth = 5
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    new_X = np.column_stack((X, labels))
    # savemarker('/media/tyh/3E5618CD561887B3/test/amop1.-1.marker',marker_r)
    a = labels
    b = np.max(a)

    for i in range(int(b) + 1):
        c = np.array(np.where(a == i))
        if c.shape[1] > 1:
            marker_r.append(cluster_centers[i, 0:3])
    marker_r = np.array(marker_r)
    return marker_r

def savetifmarker(pts, filepath):
    marker_r = np.ones((pts.shape[0], 3))
    marker_r[:, 0:2] = pts
    marker_r[:, 2] = 0
    savemarker(filepath, marker_r)


if __name__ == '__main__':
    from scipy.misc import imread
    from scipy.ndimage import gaussian_filter
    from time import time
    import matplotlib.pyplot as plt
    from io1 import savemarker, loadimg
    # from skimage.feature import peak_local_max

    # img = loadimg('/home/zyq/Documents/heatmap/test_img/'+name+'.tif')
    # img = img[:,:,0]
    # img[img>0]=1
    # label_img = np.load('/home/zyq/PycharmProjects/heatmap_detection_2d_Ori/test_img/' + name + '_1199_lab.npy')
    #    label_img1=label_img>0
    # kernel = np.ones((3, 3), np.uint8)
    # a = (np.argwhere(label_img>0.2))

    # label_img = np.asarray(label_img, dtype=np.uint8)
    # b = (np.argwhere(label_img>0.2))
    # dst = cv2.dilate(label_img,kernel)
    for k in range(99,1000,100):
        test_path = './result_chai/part_seg_attn_chai_' + str(k) + '/'
        name_list = []
        for name in os.listdir(test_path):
            if name.endswith('_chai_' + str(k) + '.npy'):
                name_list.append(name)
        for i in range(len(name_list)):
        # for i in range(9,20):
            result_img = np.load(test_path +name_list[i])
            norm_img = np.zeros(result_img.shape)
            cv2.normalize(result_img, norm_img, 0, 255, cv2.NORM_MINMAX)
            norm_img = np.asarray(norm_img, dtype=np.uint8)
            # X = np.argwhere(norm_img>150)
            # pts3 = meanshif(X)
            local_max_save = test_path + 'local_max' + str(k) + '/'
            if not os.path.exists(local_max_save):
                os.mkdir(local_max_save)

            meanshift_save_path = local_max_save

            thre = 67
            X = np.argwhere(norm_img > thre)
            pts3 = meanshif(X)
            savetifmarker(pts3, meanshift_save_path + name_list[i][:-4] + '_meanshift_' + str(thre) + '.marker')
            print(meanshift_save_path + name_list[i][:-4] + '_meanshift_' + str(thre) + '.marker' + ' is saved')

            for j in range(35,135,10):

                save_path = local_max_save
                pts4 = find_maximum(norm_img, j, True)
                pts4 = pts4+1
                savetifmarker(pts4, save_path + name_list[i][:-4] +'_' + str(j) + '.marker')
                print(save_path + name_list[i]+'_' + str(j) + '.marker'+' is saved')

            print(i)

#    savetifmarker(pts4,'/media/ttt/4CF45F63F45F4DF8/reli6.2/'+name+'_4.marker')
#    savetifmarker(coordinates,'/media/ttt/4CF45F63F45F4DF8/reli/05_t3.marker')
print('over')


# plt.imshow(norm_img, cmap='gray')
# plt.subplot(1, 2, 1)
# plt.plot(pts1[:, 1], pts1[:, 0], 'y.')
# plt.subplot(1, 2, 2)
# plt.plot(pts2[:, 1], pts2[:, 0], 'y.')
# plt.show()
