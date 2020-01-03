# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 18:14
# @Author  : Ziqi Wang
# @FileName: img_train.py
# @Email: zw280@scarletmail.rutgers.edu
import numpy as np


class NN:
    def __init__(self, kernel_size=3, lr=1e-7, input_array=None, output_array=None, method="valid"):
        self.kernel_size = kernel_size
        self.learning_rate = lr
        self.input_array = input_array
        self.output_array = output_array
        self.weights = np.random.randn(kernel_size, kernel_size)
        self.stride = 1
        self.method = method

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
        if self.method == 'SAME':
            x = np.pad(x,
                       ((self.kernel_size // 2, self.kernel_size // 2), (self.kernel_size // 2, self.kernel_size // 2)),
                       'constant', constant_values=0)
        # col_image is the gradient of the convolutional layer, we need to do sth with it
        self.col_image = self.im2col(x, self.kernel_size, self.stride)
        conv_out_c = np.dot(self.col_image, col_weights)
        # conv_out = np.reshape(conv_out, np.hstack(self.input_array.shape))
        # the output of column format
        conv_out = np.reshape(conv_out_c, np.hstack((x.shape[0] - self.kernel_size + 1,
                                                  x.shape[0] - self.kernel_size + 1)))
        return conv_out, conv_out_c

    def train(self, epoches=100):
        for t in range(epoches):
            h, h_c = self.forward(self.input_array)
            h_relu, h_c_relu = np.maximum(h, 0), np.maximum(h_c, 0)
            # out put in column format
            out_put_c = np.reshape(self.output_array, np.hstack(h_c_relu.shape))
            loss, loss_c = np.square(h_relu - self.output_array).sum(),  np.square(h_c_relu - out_put_c).sum()
            print(t, loss, loss_c)
            grad_loss, grad_loss_c = 2 * (h_relu - self.output_array), 2 * (h_c_relu - out_put_c)
            grad_h_relu, grad_h_c_relu = grad_loss.copy(), grad_loss_c.copy()
            grad_h_relu[grad_h_relu < 0] = 0
            grad_h_c_relu[grad_h_c_relu < 0] = 0
            grad_kernel_c = np.dot(self.col_image.T, grad_h_c_relu)
            grad_kernel = np.reshape(grad_kernel_c, np.hstack(self.weights.shape))
            # grad_kernel = np.dot(np.reshape(self.col_image, np.hstack(self.input_array.shape)), grad_h_relu)
            self.weights = self.weights - self.learning_rate * grad_kernel

    def predict(self, input_vector, output_vector):
        conv = self.forward(input_vector)
        conv_relu = np.maximum(conv, 0)
        loss = np.square(conv_relu - output_vector).sum()
        print(loss)


if __name__ == '__main__':
    train_input, train_output = np.random.randn(255, 255), np.random.randn(255, 255)
    # train_input, train_output = np.random.randint(0, 255, (3, 3)), np.random.randint(0, 255, (2, 2))
    # nn = NN(input_array=train_input, output_array=train_output, kernel_size=2)
    nn = NN(input_array=train_input, output_array=train_output, kernel_size=3, method='SAME')
    nn.train(1600)
    nn.predict(train_input, train_output)
