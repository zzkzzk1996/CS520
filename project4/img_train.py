# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 18:14
# @Author  : Ziqi Wang
# @FileName: img_train.py
# @Email: zw280@scarletmail.rutgers.edu
import numpy as np


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class NN:
    def __init__(self, kernel_size=3, lr=1e-6, input_array=None, output_array=None, method="valid",
                 activation_function='relu', model_path=None):
        self.kernel_size = kernel_size
        self.learning_rate = lr
        self.input_array = input_array
        self.output_array = output_array
        self.weights = self.load_model(model_path)
        self.stride = 1
        self.method = method
        self.activation = activation_function

    def load_model(self, filename):
        return np.loadtxt(filename, delimiter=",")

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
                                                     x.shape[1] - self.kernel_size + 1)))
        return conv_out, conv_out_c

    def back_conv(self, gradient_conv_c, conv_c):
        grad_kernel_c = np.dot(gradient_conv_c, conv_c)
        return np.reshape(grad_kernel_c, np.hstack(self.weights.shape))

    def train(self, epoches=100):
        for t in range(epoches):
            h, h_c = self.forward(self.input_array)
            if self.activation == 'relu':
                h_act, h_c_act = relu(h), relu(h_c)
                # out put in column format
                out_put_c = np.reshape(self.output_array, np.hstack(h_c_act.shape))
                loss, loss_c = np.square(h_act - self.output_array).sum(), np.square(h_c_act - out_put_c).sum()
                print(t, loss, loss_c)
                grad_loss, grad_loss_c = 2 * (h_act - self.output_array), 2 * (h_c_act - out_put_c)
                grad_h_act, grad_h_c_act = grad_loss.copy(), grad_loss_c.copy()
                grad_h_act[grad_h_act < 0] = 0
                grad_h_c_act[grad_h_c_act < 0] = 0
            elif self.activation == 'sigmoid':
                h_act, h_c_act = sigmoid(h), sigmoid(h_c)
                # out put in column format
                out_put_c = np.reshape(self.output_array, np.hstack(h_c_act.shape))
                loss, loss_c = np.square(h_act - self.output_array).sum(), np.square(h_c_act - out_put_c).sum()
                print(t, loss, loss_c)
                grad_loss, grad_loss_c = 2 * (h_act - self.output_array), 2 * (h_c_act - out_put_c)
                grad_h_act, grad_h_c_act = (sigmoid(grad_loss) * (1 - sigmoid(grad_loss))), (
                        sigmoid(grad_loss_c) * (1 - sigmoid(grad_loss_c)))
            elif self.activation == 'tanh':
                h_act, h_c_act = tanh(h), tanh(h_c)
                # out put in column format
                out_put_c = np.reshape(self.output_array, np.hstack(h_c_act.shape))
                loss, loss_c = np.square(h_act - self.output_array).sum(), np.square(h_c_act - out_put_c).sum()
                print(t, loss, loss_c)
                grad_loss, grad_loss_c = 2 * (h_act - self.output_array), 2 * (h_c_act - out_put_c)
                grad_h_act, grad_h_c_act = (1 - tanh(grad_loss) * tanh(grad_loss)), (
                        1 - tanh(grad_loss_c) * tanh(grad_loss_c))
            # grad_kernel_c = np.dot(self.col_image.T, grad_h_c_act)
            # grad_kernel = np.reshape(grad_kernel_c, np.hstack(self.weights.shape))
            grad_kernel = self.back_conv(self.col_image.T, grad_h_c_act)
            # grad_kernel = np.dot(np.reshape(self.col_image, np.hstack(self.input_array.shape)), grad_h_act)
            self.weights = self.weights - self.learning_rate * grad_kernel

    def predict(self, input_vector, output_vector):
        conv, conv_c = self.forward(input_vector)
        if self.activation == "relu":
            out = relu(conv)
        elif self.activation == "tanh":
            out = tanh(conv)
        elif self.activation == "sigmoid":
            out = sigmoid(conv)
        # loss = np.square(out - output_vector).sum()
        # print(loss)
        return out

    def save_model(self, file_name):
        np.savetxt(file_name, self.weights, fmt="%s", delimiter=",")


def normalization(x):
    xmin, xmax = x.min(), x.max()
    return (x - xmin) / (xmax - xmin)

# if __name__ == '__main__':
#     train_input, train_output = np.random.randn(255, 255), np.random.randn(255, 255)
#     norm_in, norm_out = normalization(train_input), normalization(train_output)
#     # train_input, train_output = np.random.randint(0, 255, (3, 3)), np.random.randint(0, 255, (2, 2))
#     # nn = NN(input_array=train_input, output_array=train_output, kernel_size=2)
#     # nn = NN(input_array=train_input, output_array=train_output, kernel_size=3, method='SAME', lr=1e-7)
#     nn = NN(input_array=norm_in, output_array=norm_out, kernel_size=3, method='SAME', activation_function='tanh', lr=1e-7)
#     # nn = NN(input_array=train_input, output_array=train_output, kernel_size=3, method='SAME',activation_function='sigmoid', lr=1e-11)
#     nn.train(200)
#     nn.predict(train_input, train_output)
print(np.random.randn(3, 3))