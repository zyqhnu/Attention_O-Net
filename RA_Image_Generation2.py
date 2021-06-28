import numpy as np
import matplotlib.pyplot as plt
import re as rere
import cv2
import os
from scipy import ndimage
from io1 import writetiff2d, savemarker
import math


def load_tiff3d(filepath):
    """Load a tiff file into 3D numpy array"""
    from libtiff import TIFF
    tiff = TIFF.open(filepath, mode='r')
    stack = []
    for sample in tiff.iter_images():
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)
    tiff.close()
    return out


def load_markera(filepath):
    """
    Load swc file as a N X x numpy array
    """
    marker = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = rere.split(',', l)
                if len(cells) > 3:
                    cells = [float(c) for c in cells[0:3]]
                    marker.append(cells)
    return np.array(marker)


def close_operation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iClose = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return iClose


heatmap_list = []
srcSegPath = './resize2_528_528_after_crop/seg_img21-40/'   # 手动分割图和marker文件都在同一个文件夹里。
save_path = './resize2_528_528_after_crop/local_seg_img21-40/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for file in os.listdir(srcSegPath):
    if file.endswith('.tif'):
        heatmap_list.append(file)

det = 6.0
sigma = det / 2

for file in heatmap_list:
    seg_img = load_tiff3d(srcSegPath + file)
    seg_img = seg_img[:, :, 0]

    marker = load_markera(srcSegPath + file + '.marker')
    marker_int = marker.astype(np.int) - 1

    # # 查看图像与marker是否匹配、对应
    # plt.figure()
    # plt.imshow(seg_img)
    # plt.plot(marker_int[:,1], marker_int[:,0], '+r')
    # plt.show()

    dist_img = ndimage.distance_transform_edt(seg_img)
    local_seg_img = np.zeros(seg_img.shape)


    for i in range(len(marker_int)):
        x = marker_int[i, 1]    # 这里x是横着，即列，y是竖着，即行；而数组是先行后列
        y = marker_int[i, 0]
        # 将坐标移动到附近邻域的最大值处
        neb = 3
        lmax_patch = dist_img[y-neb:y+neb+1, x-neb:x+neb+1]
        m, n = lmax_patch.shape
        index_1 = int(np.argmax(lmax_patch))
        yy = int(index_1 / n)
        xx = index_1 % n
        if dist_img[y+yy-3, x+xx-3]>dist_img[y,x]:
            x = x + xx - 3
            y = y + yy - 3

        marker_int[i, 1] = x
        marker_int[i, 0] = y

        print(dist_img[y, x])
        if dist_img[y, x]<det:
            base_R = max(dist_img[y, x], 1)
            R = int(2*det - base_R) + 1
            # R = int(min(R,9))

            if x-R>0 and y+R<528 and y-R>0 and x+R<528:
                one_local_seg = np.zeros(seg_img.shape)
                one_local_seg[y-R:y+R, x-R:x+R] =  seg_img[y-R:y+R, x-R:x+R]
                local_seg_img = local_seg_img + one_local_seg

            local_seg_img[local_seg_img>128] = 255
            local_seg_img[local_seg_img<129] = 0

        else:
            base_R = det
            heatmap = np.zeros(seg_img.shape, dtype=np.float)

            flipped_coords = np.asarray([y, x])
            region_start = (flipped_coords - 2 * base_R).astype(int)
            region_end = (flipped_coords + 2 * base_R / 2 + 2).astype(int)
            region_size = (region_end - region_start).astype(int)

            dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
            x_diff = dx + region_start[0] - flipped_coords[0]
            y_diff = dy + region_start[1] - flipped_coords[1]
            squared_distances = x_diff * x_diff + y_diff * y_diff
            cropped_heatmap = 1 * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
            heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1]] = cropped_heatmap[:, :]
            heatmap[heatmap > 0.05] = 255
            heatmap[heatmap < 0.05] = 0

            local_seg_img = local_seg_img + heatmap
            zz = 1
            # # 膨胀一下
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # dilate = cv2.dilate(local_seg_img, kernel, iterations=1)

            # local_seg_img = local_patch(x,y,R,seg_img,dist_img)

    Ori_size = local_seg_img.shape
    marker_int[:,0] = marker_int[:,0] * 512.0/Ori_size[0]
    marker_int[:,1] = marker_int[:,1] * 512.0/Ori_size[1]
    # marker[:,2] = marker[:,2] * 1/Ori_size[2];
    crop_size = (512, 512)
    img_512 = cv2.resize(local_seg_img, crop_size, interpolation = cv2.INTER_CUBIC)
    img_512[img_512 > 128] = 255
    img_512[img_512 < 129] = 0
    plt.figure()
    plt.imshow(img_512)
    plt.plot(marker_int[:,1], marker_int[:,0], '+r')
    plt.show()

    # writetiff2d(save_path + file, dilate.astype(np.uint8))
    writetiff2d(save_path + file, img_512.astype(np.uint8))

    marker_int = np.array(list(set([tuple(t) for t in marker_int])))
    savemarker(save_path + file[:-4]+'_manual1_B_Cr_helan_Center_512_512.marker', marker_int)

print('over')
