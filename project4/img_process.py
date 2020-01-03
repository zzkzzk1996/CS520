# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 1:50
# @Author  : Ziqi Wang
# @FileName: img_process.py
# @Email: zw280@scarletmail.rutgers.edu
from skimage import io, data, color, img_as_ubyte
import numpy as np
from img_train import NN


def read_img(path):
    return io.imread(path)


def rgb2lab(img):
    return color.rgb2lab(img)


def rgb2grey(img):
    return color.rgb2grey(img)


def get_train_data(path):
    img = io.imread(path)
    img_shape = img.shape
    x = color.rgb2lab((1.0 / 255 * img))[:, :, 0]
    y = color.rgb2lab((1.0 / 255 * img))[:, :, 1:]
    # normalize x, y
    y /= 128
    return x, y, img_shape


img = read_img('D:\\CS520\\project4\\imgs\\beach01.jpg')
# print(img)
# print(rgb2grey(img))
# plt.imshow(img)
# print(img.shape)
in_arr, out_arr, in_shape = get_train_data('D:\\CS520\\project4\\imgs\\beach01.jpg')

nn_1 = NN(input_array=in_arr, output_array=out_arr[:, :, 0], kernel_size=3, method='SAME',
          activation_function='tanh', lr=1e-12, model_path='nn_1.txt')
nn_2 = NN(input_array=in_arr, output_array=out_arr[:, :, 1], kernel_size=3, method='SAME',
          activation_function='tanh', lr=1e-12, model_path='nn_2.txt')

nn_1.train(6000)
nn_2.train(6000)
nn_1.save_model('nn_1.txt')
nn_2.save_model('nn_2.txt')
y1, y2 = nn_1.predict(in_arr, out_arr[:, :, 0]) * 128, nn_2.predict(in_arr, out_arr[:, :, 1]) * 128
tmp = np.zeros(in_shape)
tmp[:, :, 0] = in_arr
tmp[:, :, 1] = y1
tmp[:, :, 2] = y2
io.imsave("D:\\CS520\\project4\\imgs\\test_image_result.png", img_as_ubyte(color.lab2rgb(tmp)))
# io.imsave("test_image_gray.png", color.rgb2gray(lab2rgb(tmp)))