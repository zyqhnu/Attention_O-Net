import os
from Attention_O_Net_architecture import AONetJunctionDetectionModule

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_data_dir = './256_256/img/'
img_part_seg_dir = './256_256/part_seg/'
label_data_dir = './256_256/label/'

image_paths = []
image_part_seg_paths = []
label_paths = []

for case in os.listdir(img_data_dir):
        image_paths.append(os.path.join(img_data_dir,case))
for case in os.listdir(img_part_seg_dir):
        image_part_seg_paths.append(os.path.join(img_part_seg_dir,case))
for case in os.listdir(label_data_dir):
        label_paths.append(os.path.join(label_data_dir,case))

detection = AONetJunctionDetectionModule(256, 256, channels=1, numclass=2, numheartmap=1, costname=('entry_loss', 'mse',))
detection.train(image_paths, image_part_seg_paths, label_paths,  './model/part_seg_attn_7.30.cpkt',
                "entry_", 0.001, 0.5, 3000, 1, [6, 7], model_continue= None)   # 执行train方法

