#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xlwt
import numpy as np
import tensorflow as tf
from layer import (conv2d, deconv2d, normalizationlayer2d, crop_and_concat2d, resnet_Add, weight_xavier_init,
                   bias_variable, save_images, sepconv2d)
from io1 import normalize, dataset_normalized


def conv_bn_relu_drop(x, kernal, phase, drop, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=[kernal[0],kernal[1],kernal[2],4], n_inputs=kernal[0] * kernal[1] * kernal[2],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        PW = weight_xavier_init(shape=[1,1,4*kernal[2],kernal[3]], n_inputs=kernal[0] * kernal[1] * kernal[2],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_PW')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = sepconv2d(x, W, PW) + B
        conv = normalizationlayer2d(conv, is_train=phase, height=height, width=width, norm_type='group',
                                    scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def down_sampling(x, kernal, phase, drop, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv2d(x, W, 2) + B
        conv = normalizationlayer2d(conv, is_train=phase, height=height, width=width, norm_type='group',
                                    scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv2d(x, W, samefeture) + B
        conv = tf.nn.relu(conv)
        return conv


def conv_sigmod(x, kernal, scope=None, activeflag=True):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv2d(x, W) + B
        if activeflag:
            conv = tf.nn.sigmoid(conv)
        return conv


# Attention module
def attn(seg_layer, heatmap_layer, inputfilters, outfilters, phase, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        kernal1 = (1, 1, inputfilters, inputfilters)
        kernalx = (1, 1, inputfilters, outfilters)
        seg_layer_att = conv_relu(seg_layer, kernal=kernal1, scope=scope + 'conv1_relu')
        seg_layer_att = conv_sigmod(seg_layer_att, kernalx, scope=scope + 'conv2_sigmd', activeflag=True)
        assert seg_layer_att.get_shape().as_list()==heatmap_layer.get_shape().as_list(), '注意力模块 输入维度与卷积后维度不匹配 不能相乘和相加'

        seg_layer_att = tf.multiply(seg_layer_att, heatmap_layer)
        seg_layer_att = resnet_Add(x1=seg_layer_att, x2=heatmap_layer)
        # seg_layer_att = normalizationlayer2d(seg_layer_att, is_train=phase, height=height, width=width, norm_type='group',
        #                             scope=scope)
        return seg_layer_att


def conv_relu(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv2d(x, W) + B
        conv = tf.nn.relu(conv)
        return conv


# Attention O-Net architecture:
def _create_AO_net(X, image_width, image_height, image_channel, phase, drop, n_class=1, n_heartmap=1):   # n_class = 2
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, image_channel, 64), phase=phase, drop=drop, scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='down1')

    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)

    layer2_seg = conv_bn_relu_drop(x=down1, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer2_1_seg')
    layer2_seg = conv_bn_relu_drop(x=layer2_seg, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer2_2_seg')
    layer2_seg = resnet_Add(x1=down1, x2=layer2_seg)

    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='down2')
    down2_seg = down_sampling(x=layer2_seg, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='down2_seg')

    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer3_2')
    layer3 = resnet_Add(x1=down2, x2=layer3)

    layer3_seg = conv_bn_relu_drop(x=down2_seg, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer3_1_seg')
    layer3_seg = conv_bn_relu_drop(x=layer3_seg, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='layer3_2_seg')
    layer3_seg = resnet_Add(x1=down2_seg, x2=layer3_seg)

    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 64, 128), phase=phase, drop=drop, scope='down3')
    down3_seg = down_sampling(x=layer3_seg, kernal=(3, 3, 64, 128), phase=phase, drop=drop, scope='down3_seg')

    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer4_2')
    layer4 = resnet_Add(x1=down3, x2=layer4)

    layer4_seg = conv_bn_relu_drop(x=down3_seg, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer4_1_seg')
    layer4_seg = conv_bn_relu_drop(x=layer4_seg, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer4_2_seg')
    layer4_seg = resnet_Add(x1=down3_seg, x2=layer4_seg)

    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='down4')
    down4_seg = down_sampling(x=layer4_seg, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='down4_seg')

    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer5_2')
    layer5 = resnet_Add(x1=down4, x2=layer5)

    layer5_seg = conv_bn_relu_drop(x=down4_seg, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer5_1_seg')
    layer5_seg = conv_bn_relu_drop(x=layer5_seg, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='layer5_2_seg')
    layer5_seg = resnet_Add(x1=down4_seg, x2=layer5_seg)

    # layer->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 128, 128), samefeture=True, scope='deconv1')
    deconv1_seg = deconv_relu(x=layer5_seg, kernal=(3, 3, 128, 128), samefeture=True, scope='deconv1_seg')

    # layer8->convolution
    layer6 = crop_and_concat2d(layer4, deconv1)
    _, H, W, _ = layer4.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 256, 128), height=H, width=W, phase=phase, drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 128, 128), height=H, width=W, phase=phase, drop=drop, scope='layer6_2')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)

    layer6_seg = crop_and_concat2d(layer4_seg, deconv1_seg)
    _, H_seg, W_seg, _ = layer4_seg.get_shape().as_list()
    layer6_seg = conv_bn_relu_drop(x=layer6_seg, kernal=(3, 3, 256, 128), height=H_seg, width=W_seg, phase=phase, drop=drop, scope='layer6_1_seg')
    layer6_seg = conv_bn_relu_drop(x=layer6_seg, kernal=(3, 3, 128, 128), height=H_seg, width=W_seg, phase=phase, drop=drop, scope='layer6_2_seg')
    layer6_seg = resnet_Add(x1=deconv1_seg, x2=layer6_seg)

    attn1 = attn(layer6_seg, layer6, 128, 128, phase=phase, scope='attn1')

    # layer9->deconvolution  这里是第三个数字是输出，第四个是输入
    deconv2 = deconv_relu(x=attn1, kernal=(3, 3, 64, 128), samefeture=False, scope='deconv2')
    deconv2_seg = deconv_relu(x=layer6_seg, kernal=(3, 3, 64, 128), samefeture=False, scope='deconv2_seg')

    # layer8->convolution
    layer7 = crop_and_concat2d(layer3, deconv2)
    _, H, W, _ = layer3.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 128, 64), height=H, width=W, phase=phase,drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 64, 64), height=H, width=W, phase=phase,drop=drop, scope='layer7_2')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)

    layer7_seg = crop_and_concat2d(layer3_seg, deconv2_seg)
    _, H_seg, W_seg, _ = layer3_seg.get_shape().as_list()
    layer7_seg = conv_bn_relu_drop(x=layer7_seg, kernal=(3, 3, 128, 64), height=H_seg, width=W_seg, phase=phase,drop=drop, scope='layer7_1_seg')
    layer7_seg = conv_bn_relu_drop(x=layer7_seg, kernal=(3, 3, 64, 64), height=H_seg, width=W_seg, phase=phase,drop=drop, scope='layer7_2_seg')
    layer7_seg = resnet_Add(x1=deconv2_seg, x2=layer7_seg)

    attn2 = attn(layer7_seg, layer7, 64, 64, phase=phase, scope='attn2')

    # layer9->deconvolution
    deconv3 = deconv_relu(x=attn2, kernal=(3, 3, 64, 64), samefeture=True, scope='deconv3')
    deconv3_seg = deconv_relu(x=layer7_seg, kernal=(3, 3, 64, 64), samefeture=True, scope='deconv3_seg')

    # layer8->convolution
    layer8 = crop_and_concat2d(layer2, deconv3)
    _, H, W, _ = layer2.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 128, 64), height=H, width=W, phase=phase,drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 64, 64), height=H, width=W, phase=phase,drop=drop, scope='layer8_2')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)

    layer8_seg = crop_and_concat2d(layer2_seg, deconv3_seg)
    _, H_seg, W_seg, _ = layer2_seg.get_shape().as_list()
    layer8_seg = conv_bn_relu_drop(x=layer8_seg, kernal=(3, 3, 128, 64), height=H_seg, width=W_seg, phase=phase, drop=drop, scope='layer8_1_seg')
    layer8_seg = conv_bn_relu_drop(x=layer8_seg, kernal=(3, 3, 64, 64), height=H_seg, width=W_seg, phase=phase, drop=drop, scope='layer8_2_seg')
    layer8_seg = resnet_Add(x1=deconv3_seg, x2=layer8_seg)

    attn3 = attn(layer8_seg, layer8, 64, 64, phase=phase, scope='attn3')

    # layer9->deconvolution
    deconv4 = deconv_relu(x=attn3, kernal=(3, 3, 64, 64), samefeture=True, scope='deconv4')
    deconv4_seg = deconv_relu(x=layer8_seg, kernal=(3, 3, 64, 64), samefeture=True, scope='deconv4_seg')

    # layer8->convolution
    layer9 = crop_and_concat2d(layer1, deconv4)
    layer9_seg = crop_and_concat2d(layer1, deconv4_seg)

    _, H, W, _ = layer1.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 128, 64), height=H, width=W, phase=phase, drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 64, 64), height=H, width=W, phase=phase, drop=drop, scope='layer9_2')

    _, H_seg, W_seg, _ = layer1.get_shape().as_list()
    layer9_seg = conv_bn_relu_drop(x=layer9_seg, kernal=(3, 3, 128, 64), height=H_seg, width=W_seg, phase=phase, drop=drop, scope='layer9_1_seg')
    layer9_seg = conv_bn_relu_drop(x=layer9_seg, kernal=(3, 3, 64, 64), height=H_seg, width=W_seg, phase=phase, drop=drop, scope='layer9_2_seg')

    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    layer9_seg = resnet_Add(x1=deconv4_seg, x2=layer9_seg)

    output_map_logit1 = conv_sigmod(x=layer9_seg, kernal=(1, 1, 64, n_class), scope='output1', activeflag=False)

    attn4 = attn(layer9_seg, layer9, 64, 64, phase=phase, scope='attn4')
    output_map_logit2 = conv_sigmod(x=attn4, kernal=(1, 1, 64, n_heartmap), scope='output2', activeflag=False)

    return output_map_logit1, output_map_logit2


# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class AONetJunctionDetectionModule(object):

    def __init__(self, image_height, image_width, channels=1, numclass=2, numheartmap=1,costname=("",),
                 inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.numclass = numclass    # 2
        self.numheartmap = numheartmap
        self.labelchannels = numclass   # 2
        self.dimension = 2

        self.X = tf.placeholder("float", shape=[None, self.image_height, self.image_width, self.channels])
        self.Y_gt1 = tf.placeholder("int32", shape=[None, self.image_height, self.image_width])
        self.Y_gt2 = tf.placeholder("float", shape=[None, self.image_height, self.image_width, self.numheartmap])
        self.lr = tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)
        self.drop = tf.placeholder('float')

        self.Y_pred_logit1, self.Y_pred_logit2 = _create_AO_net(self.X, self.image_height, self.image_width,
                                                 self.channels, self.phase, self.drop, self.numclass, self.numheartmap)
        self.cost = self.__get_cost(self.Y_pred_logit1, self.Y_gt1, costname[0]) + 3*self.__get_cost(self.Y_pred_logit2,
                                    self.Y_gt2, costname[1])

        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            self.Y_pred_logit1 = tf.nn.softmax(self.Y_pred_logit1)
            saver.restore(self.sess, model_path)
            print("restored")


    def __get_cost(self, Y_pred, Y_gt, cost_name):
        if cost_name == "L2-loss":
            loss = tf.nn.l2_loss(Y_pred - Y_gt)
            return loss
        if cost_name == "entry_loss":
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_gt, logits=Y_pred)
            loss = tf.reduce_mean(loss)
            return loss
        if cost_name == "mse":
            loss = tf.losses.mean_squared_error(Y_gt, Y_pred)
            return loss


    def __get_landmark(self, image):
        max_index = np.argmax(image)
        coord = np.array(np.unravel_index(max_index, dims=image.shape), np.int)
        value = image[tuple(coord)]
        return coord, value


    def __loadnumtraindata(self, train_images, image_part_seg_paths, train_lanbels, num_sample):
        subbatch_xs = np.empty((num_sample, self.image_height, self.image_width, self.channels))
        subbatch_ys1 = np.empty((num_sample, self.image_height, self.image_width))
        subbatch_ys2 = np.empty((num_sample, self.image_height, self.image_width, self.numheartmap))

        for num in range(len(train_images)):
            image = np.load(train_images[num])
            image_part_seg = np.load(image_part_seg_paths[num])
            labels = np.load(train_lanbels[num])

            heart_label=labels
            # label = np.zeros((self.image_height, self.image_width))
            # label[labels>0.01]=1
            label = image_part_seg

            subbatch_xs[num, :, :, :] = np.reshape(image, (self.image_height, self.image_width, self.channels))
            subbatch_ys1[num, :, :] = np.reshape(label, (self.image_height, self.image_width))
            subbatch_ys2[num, :, :, :] = np.reshape(heart_label, (self.image_height, self.image_width, self.numheartmap))
        permutation = np.random.permutation(num_sample)
        subbatch_xs = subbatch_xs[permutation, :, :, :]
        subbatch_ys1 = subbatch_ys1[permutation, :, :]
        subbatch_ys2 = subbatch_ys2[permutation, :, :, :]
        subbatch_xs = subbatch_xs.astype(np.float)
        subbatch_ys1 = subbatch_ys1.astype(np.int)
        return subbatch_xs, subbatch_ys1, subbatch_ys2

    def train(self, train_images,image_part_seg_paths, train_labels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=5, batch_size=1, showwind=[8, 8], model_continue=None):
        num_sample = len(train_images)
        train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.cost)
#        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
#        AdamOptimizer
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=500)

        tf.summary.scalar("loss", self.cost)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
#        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        if model_continue!=None:
            saver.restore(sess, model_continue)

        xs, ys1, ys2 = self.__loadnumtraindata(train_images, image_part_seg_paths, train_labels, num_sample)

        workbook = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = workbook.add_sheet('test', cell_overwrite_ok=True)

        for i in range(train_epochs):
            # Extracting num_sample images and labels from given data
#            if i % num_sample == 0 or i == 0:
            bs = 0
            be = 0
            fflag=0
            while bs < len(train_images):

                be = bs + batch_size if bs+batch_size <len(train_images) else len(train_images)
                batch_xs = xs[bs:be,:,:,:]
                batch_ys1 = ys1[bs:be,:,:]
                batch_ys2 = ys2[bs:be,:,:, :]

                bs = be
            # get new batch
#            batch_xs, batch_ys, index_in_epoch = _next_batch(subbatch_xs, subbatch_ys, batch_size, index_in_epoch)
            # Extracting images and labels from given data
                batch_xs = batch_xs.astype(np.float)
                batch_ys1 = batch_ys1.astype(np.int)
                batch_ys2 = batch_ys2.astype(np.float)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
#            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss,_ = sess.run([self.cost,train_op], feed_dict={self.X: batch_xs,
                                                            self.Y_gt1: batch_ys1,
                                                            self.Y_gt2: batch_ys2,
                                                            self.lr: learning_rate,
                                                            self.phase: 1,
                                                            self.drop: dropout_conv})
                sheet.write(i, fflag, float(train_loss))
                fflag+=1
                print('epochs %d training_loss  => %.5f ' % (i, train_loss))

            if(((i+1)%20)==0):
                save_path = saver.save(sess, model_path,global_step=i)
                print("Model saved in file:", save_path)
                workbook.save(model_path+'excelFile.xls')
        workbook.save(model_path+'excelFile.xls')

    def prediction(self, test_images):
        assert self.image_width == test_images.shape[1], \
            'prediction process the input size is not equal vnet input size'
        test_images = np.reshape(test_images, (self.image_height, self.image_width, self.channels))
        y_dummy1 = np.zeros(shape=(self.image_height, self.image_width))
        y_dummy2 = np.zeros(shape=(self.image_height, self.image_width, self.numheartmap))
        test_images = test_images.astype(np.float)
        pred1, pred2 = self.sess.run([self.Y_pred_logit1, self.Y_pred_logit2],
                                                feed_dict={self.X: [test_images],
                                                           self.Y_gt1: [y_dummy1],
                                                           self.Y_gt2: [y_dummy2],
                                                           self.phase: 1,
                                                           self.drop: 1})

        result1=np.squeeze(pred1)
        result1=np.argmax(result1,axis=2)
        result2 = pred2.astype(np.float)

        result1 = np.reshape(result1, (self.image_height, self.image_width))
        result2 = np.reshape(result2, (self.image_height, self.image_width, self.numheartmap))
        return result1, result2

    def inference(self, img):

        input_array = dataset_normalized(img)  # normalize image to mean 0 std 1 , limit to 0 and 1
        # input_array = normalize(img)  # normalize image to mean 0 std 1
        labelmap, heatmaps_array = self.prediction(input_array)
        heatmaps_array=np.squeeze(heatmaps_array)
        return labelmap, heatmaps_array