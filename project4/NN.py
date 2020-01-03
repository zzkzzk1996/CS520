# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 15:56
# @Author  : Ziqi Wang
# @FileName: NN.py
# @Email: zw280@scarletmail.rutgers.edu
# simple NN, the loss is very large
import numpy as np


def relu(x):
    return np.maximum(0, x)


class NN:
    def __init__(self, input_width=255, hidden_width=100, output_width=255, lr=0.05, input_array=None,
                 output_array=None):
        self.hidden_layer = np.random.randn(input_width, hidden_width)
        self.output = np.random.randn(hidden_width, output_width)
        self.learning_rate = lr
        self.input_array = input_array
        self.output_array = output_array

    def train(self, epoches=500):
        for t in range(epoches):
            h = self.input_array.dot(self.hidden_layer)
            h_relu = relu(h)
            predict = h_relu.dot(self.output)
            loss = np.square(predict - self.output_array).sum()
            print(t, loss)
            # the gradient of the loss
            grad_predict = 2.0 * (predict - self.output_array)
            # the gradient of the output layer is h_relu, according to dwx/dx = w
            # backpropagation
            grad_output = h_relu.T.dot(grad_predict)
            # the order is due to the matrix order
            grad_h_relu = grad_predict.dot(self.output.T)
            grad_h = grad_h_relu.copy()
            grad_h[grad_h < 0] = 0
            # as above
            grad_hidden = self.input_array.T.dot(grad_h)

            self.output -= self.learning_rate * grad_output
            self.hidden_layer -= self.learning_rate * grad_hidden

    def predict(self, input_vector, output_vector):
        h = input_vector.dot(self.hidden_layer)
        h_relu = relu(h)
        predict = h_relu.dot(self.output)
        loss = np.square(predict - output_vector).sum()
        print(loss)


def im2col(image, ksize, stride):
    # image is a 2d tensor([width ,height])
    image_col = []
    for i in range(0, image.shape[0] - ksize + 1, stride):
        for j in range(0, image.shape[1] - ksize + 1, stride):
            col = image[i:i + ksize, j:j + ksize].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col

# test fucntion
if __name__ == '__main__':
    train_input, train_output = np.random.randn(255, 255), np.random.randn(255, 255)
    test_input, test_output = np.random.randn(255, 255), np.random.randn(255, 255)
    # train_input = im2col(train_input, 3, 1)
    nn = NN(input_array=train_input, output_array=train_output, lr=1e-7)
    nn.train(1000000)
    nn.predict(test_input, test_output)
