# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 17:45
# @Author  : Ziqi Wang
# @FileName: CNN.py
# @Email: zw280@scarletmail.rutgers.edu
# from :https://zhuanlan.zhihu.com/p/40951745
import numpy as np


def forward(self, x):
    col_weights = self.weights.reshape([-1, self.output_channels])
    if self.method == 'SAME':
        x = np.pad(x, (
            (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                   'constant', constant_values=0)

    self.col_image = im2col(x, self.ksize, self.stride)
    conv_out = np.dot(self.col_image, col_weights) + self.bias
    conv_out = np.reshape(conv_out, np.hstack(([self.batchsize], self.eta[0].shape)))
    return conv_out


def gradient(self, eta):
    self.eta = eta
    col_eta = np.reshape(eta, [-1, self.output_channels])
    self.w_gradient = np.dot(self.col_image.T,
                             col_eta).reshape(self.weights.shape)
    self.b_gradient = np.sum(col_eta, axis=0)

    # deconv of padded eta with flippd kernel to get next_eta
    if self.method == 'VALID':
        pad_eta = np.pad(self.eta, (
            (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                         'constant', constant_values=0)

    if self.method == 'SAME':
        pad_eta = np.pad(self.eta, (
            (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                         'constant', constant_values=0)

    flip_weights = self.weights[::-1, ...]
    flip_weights = flip_weights.swapaxes(1, 2)
    col_flip_weights = flip_weights.reshape([-1, self.input_channels])

    col_pad_eta = im2col(pad_eta, self.ksize, self.stride)
    next_eta = np.dot(col_pad_eta, col_flip_weights)
    next_eta = np.reshape(next_eta, self.input_shape)
    return next_eta


def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for b in range(image.shape[0]):
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[b, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
    image_col = np.array(image_col)

    return image_col
