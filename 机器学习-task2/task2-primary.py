# 导入库
import numpy as np
# 导入random模块
import random
from tensorflow.keras.datasets import mnist

# 下载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
# 将图像数据展平为一维向量
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))
# 将像素值缩放到0到1之间
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# 将标签进行独热编码
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]


class Network(object):
    def __init__(self, sizes):
        # 获取列表长度，即神经网络的层数
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 生成随机数组：用列表推导式创建一个列表，其中每个元素都是一个形状为 (y, 1) 的随机数数组，这些数组表示每一层的偏置项。
        # sizes[1:] 表示从第*二*层开始到最后一层的神经元数量。这是因为偏置项的数量与除输入层以外的每一层的神经元数量相对应
        # 按照惯例，假设第一层是输入层，我们不会为这些神经元设置任何偏置项，因为偏置项仅在计算后续层的输出时使用。
        self.biases=[np.random.randn(y,1) for y in sizes [1:]]
        # 初始化权重
        # sizes[:-1] 表示从 sizes 列表中取出除了最后一个元素外的所有元素，即表示输入层到倒数第二层的神经元数量。
        # zip(sizes[:-1], sizes[1:]) 将输入层到倒数第二层的神经元数量和第二层到输出层的神经元数量一一对应地组合成一个元组。
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            # 用激活函数创建一个激活值
            # np.dot(w, a)执行了权重矩阵w与上一层的输出或激活值a的矩阵乘法操作，得到一个中间结果向量。然后，该中间结果向量与偏置向量 b 相加，得到一个新的向量。
            # 接下来，通过 sigmoid 函数对这个新向量的每个元素进行激活操作，最终得到一个激活值向量
            a = sigmoid(np.dot(w, a) + b)
        return a

    # 创建SGD（随机梯度下降）优化器
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # 如果提供了 test_data，则将测试数据集的长度存储在变量 n_test 中。
        # 目的是获取训练数据和测试数据的样本数量，以便在训练过程中进行进度跟踪和评估，提高正确率。
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        # 外层是数据集的迭代（epoch）
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # 内层是随机梯度下降算法中小批量集合的迭代，每个批量（batch）都会计算一次梯度，进行一次全体参数的更新（一次更新就是一个step）：
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # 评估模型性能：其中 j 表示当前周期的索引，n_test 是测试数据集的样本数量。输出的字符串格式为 "Epoch {当前周期}: {测试数据集上的性能} / {测试样本数量}
            # 其实可以删去
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


    def update_mini_batch(self,mini_batch, eta):
        # 应用反向传播的梯度下降算法，对单个小批量数据进行更新神经网络的权重和偏置。
        # eta 是学习率。

        # 首先初始化了两个列表 nabla_b 和 nabla_w，用于存储权重和偏置的梯度。
        # nabla_b 列表中的每个元素都是一个与相应偏置形状相同的全零数组，nabla_w 列表中的每个元素都是一个与相应权重形状相同的全零数组。
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 使用 self.backprop(x, y) 方法计算梯度 delta_nabla_b 和 delta_nabla_w。
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 更新神经网络的权重和偏置
        # 更新的规则是减去学习率 eta 除以小批量数据集的大小 len(mini_batch)，再乘以梯度 nw。
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        # 反向传播算法
        # 先初始化两个列表，再将样本赋值给activation（当前的激活层）
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
    # 输入样本X添加到列表activation中作为第一层的激活值
        activations = [x]
        # 将计算得到的加权输入z添加到列表zs中
        zs = []
        # 遍历偏置和权重
        for b, w in zip(self.biases, self.weights):
            # 通过矩阵乘法 np.dot(w, activation) 将权重 w 和上一层的激活值 activation 相乘，然后加上偏置 b。这样就得到了该层的加权输入 z。
            z = np.dot(w, activation) + b
            # 将z添加到列表zs中
            zs.append(z)
            # 将z传入激活函数
            activation = sigmoid(z)
            # 将activation添加到activations中
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                    for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

# Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))