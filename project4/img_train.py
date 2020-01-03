# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 18:14
# @Author  : Ziqi Wang
# @FileName: img_train.py
# @Email: zw280@scarletmail.rutgers.edu
import numpy as np


class NN:
    def __init__(self, kernel_size=3, lr=0.005, input_array=None, output_array=None):
        self.kernel_size = kernel_size
        self.learning_rate = lr
        self.input_array = input_array
        self.output_array = output_array
        self.weights = np.random.randn(kernel_size, kernel_size)
        self.stride = 1

    def im2col(self, input_img, ksize, stride):
        # input_img is a 2d tensor([width ,height])
        image_col = []
        for i in range(0, input_img.shape[0] - ksize + 1, stride):
            for j in range(0, input_img.shape[1] - ksize + 1, stride):
                col = input_img[i:i + ksize, j:j + ksize].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)
        return image_col

    def forward(self, x):
        col_weights = self.weights.reshape([-1])
        x = np.pad(x, ((self.kernel_size // 2, self.kernel_size // 2), (self.kernel_size // 2, self.kernel_size // 2)),
                   'constant', constant_values=0)
        # col_image is the gradient of the convolutional layer
        self.col_image = self.im2col(x, self.kernel_size, self.stride)
        conv_out = np.dot(self.col_image, col_weights)
        conv_out = np.reshape(conv_out, np.hstack(self.input_array.shape))
        return conv_out

    def train(self, epoches=100):
        for t in range(epoches):
            h = self.forward(self.input_array)
            h_relu = np.maximum(h, 0)
            loss = np.square(h_relu - self.output_array).sum()
            print(t, loss)
            grad_loss = 2 * (h_relu - self.output_array)
            grad_h_relu = grad_loss.copy()
            grad_h_relu[grad_h_relu < 0] = 0
            grad_kernel = np.dot(np.reshape(self.col_image, np.hstack(self.input_array.shape)), grad_h_relu)
            self.weights -= self.learning_rate * grad_kernel

    def predict(self, input_vector, output_vector):
        conv = self.forward(input_vector)
        conv_relu = np.maximum(conv, 0)
        loss = np.square(conv_relu - output_vector).sum()
        print(loss)


if __name__ == '__main__':
    train_input, train_output = np.random.randint(0, 255, (255, 255)), np.random.randint(0, 255, (255, 255))
    nn = NN(input_array=train_input, output_array=train_output)
    nn.train(500)
    nn.predict(train_input, train_output)
