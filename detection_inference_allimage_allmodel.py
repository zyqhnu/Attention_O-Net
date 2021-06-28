from __future__ import print_function, division
import os
import matplotlib.pylab as plt
import numpy as np
from io1 import writetiff3d, savemarker, loadimg
from Attention_O_Net_architecture import AONetJunctionDetectionModule
import cv2
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_path22 = './rensponse_test/DRIVE_green_512/'
# test_img_list = ['01.tif', '02.tif', '03.tif']
test_img_list = []
for file in os.listdir(test_path22):
    if file.endswith('.tif'):
        test_img_list.append(file)

for j in range(99,200,100):
    model_path = './model/part_seg_attn_double_8_8.cpkt-' + str(j)

    detection = AONetJunctionDetectionModule(512, 512, channels=1, numclass=2, numheartmap=1, costname=('entry_loss',
                                             'mse'), inference=True, model_path=model_path)
    npy_save_path = './' + 'part_seg_attn_double_' + str(j) + '/'
    if not os.path.exists(npy_save_path):
        os.mkdir(npy_save_path)
        print(npy_save_path + '不存在，已创建')

    for i in range(len(test_img_list)):
        name = test_img_list[i]
        test_path_name = test_path22 + name
        img = loadimg(test_path_name)

        # old_time = datetime.datetime.now()
        result1, result2=detection.inference(img)
        # new_time = datetime.datetime.now()
        # print('时间间隔： %s', (new_time-old_time))

        # result2[result2<0.08]=0

        npy_save_name = name[:-4]+'part_seg_attn_double_' + str(j) + '.npy'
        np.save(npy_save_path + npy_save_name, result2)
        print(npy_save_path + npy_save_name + ' is saved')

        # norm_img = np.zeros(result2.shape)
        # cv2.normalize(result2, norm_img, 0, 255, cv2.NORM_MINMAX)
        # norm_img = np.asarray(norm_img, dtype=np.uint8)
        # heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        # org_img = img[:,:,0]
        # org_img = np.expand_dims(org_img, -1)
        # org_img = np.concatenate((org_img, org_img, org_img), axis=-1)
        # org_img = np.asarray(org_img, dtype=np.uint8)
        # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        #
        # img_add = cv2.addWeighted(org_img, 0.5, heat_img, 0.5, 0)
        # if not os.path.exists('./jpg/'):
        #     os.mkdir('./jpg/')
        #     print('./jpg/ ' + '不存在，已创建')
        # jpg_name = './jpg/' + name[:-4] + 'part_seg_attn_double' + str(j) + '.jpg'
        # cv2.imwrite(jpg_name, img_add)  # 把img变量保存到img.png,图片品质为70

    from keras import  backend as k
    k.clear_session()

print('over')
