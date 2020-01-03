# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 6:11
# @Author  : Ziqi Wang
# @FileName: classify.py
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
    x = color.rgb2lab(img)[:, :, 0]
    y = color.rgb2lab(img)[:, :, 1:]
    # normalize x, y

    return x, y, img_shape


def take_divide(matrix):
    # divide the color space to [0, 1, 2, 3, 4]
    matrix[np.where(matrix < - 64)] = -2.0
    matrix[np.where((-64 <= matrix) & (matrix < 0))] = -1.0
    matrix[np.where((0 <= matrix) & (matrix < 64))] = 1.0
    matrix[np.where((64 <= matrix) & (matrix < 128))] = 2.0
    return matrix


# img = read_img('D:\\CS520\\project4\\imgs\\sunset01.jpg')
# print(img)
# print(rgb2grey(img))
# plt.imshow(img)
# print(img.shape)
in_arr, out_arr, in_shape = get_train_data('D:\\CS520\\project4\\imgs\\sunset01.jpg')
out_array = take_divide(out_arr)
nn_3 = NN(input_array=in_arr, output_array=out_arr[:, :, 0], kernel_size=3, method='SAME',
          activation_function='tanh', lr=1e-11, model_path='nn_3.txt')
nn_4 = NN(input_array=in_arr, output_array=out_arr[:, :, 1], kernel_size=3, method='SAME',
          activation_function='tanh', lr=1e-11, model_path='nn_4.txt')

nn_3.train(50)
nn_4.train(50)
nn_3.save_model('nn_3.txt')
nn_4.save_model('nn_4.txt')
y1, y2 = nn_3.predict(in_arr, out_arr[:, :, 0]) * 64, nn_3.predict(in_arr, out_arr[:, :, 1]) * 64
tmp = np.zeros(in_shape)
tmp[:, :, 0] = in_arr
tmp[:, :, 1] = y1
tmp[:, :, 2] = y2
io.imsave("D:\\CS520\\project4\\imgs\\test_image_result_1.png", img_as_ubyte(color.lab2rgb(tmp)))
# io.imsave("test_image_gray.png", color.rgb2gray(lab2rgb(tmp)))
