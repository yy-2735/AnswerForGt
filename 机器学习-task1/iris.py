import numpy as np
# numpy 是 Python 中用于科学计算的一个重要库，提供了高性能的多维数组对象和各种用于数组操作的函数。
# as np 是给导入的 NumPy 库起一个别名
import torch
# PyTorch 是一个用于构建深度学习模型的开源机器学习库。它提供了丰富的工具和函数，用于创建和训练神经网络，并在各种任务中进行深度学习研究和开发。
from sklearn import datasets
# 是 Python 中从 scikit-learn 库导入 datasets 模块的语句。
# scikit-learn 是一个流行的机器学习库，提供了许多用于数据预处理、模型选择、模型评估和数据可视化的工具和函数。
# datasets 模块是 scikit-learn 库中的一个子模块，用于加载和生成常用的示例数据集。
import torch.nn as nn
from sklearn.model_selection import train_test_split
# nn 模块是一个核心模块，提供了用于构建神经网络的各种类和函数,可用类和函数来创建神经网络模型的各个组件。
import matplotlib.pyplot as plt

# matplotlib 是一个用于数据可视化的流行的Python库。
# pyplot 模块是 matplotlib 库中的一个子模块，提供了类似于 MATLAB 的绘图接口，用于创建各种类型的图表和图形。

# 获得数据集:datasets用于生成示例数据集，load_iris() 是 datasets 模块中的一个函数，它用于加载**鸢尾花**数据集。
dataset = datasets.load_iris()
# 输出数据集的特征矩阵
print("特征矩阵:")
print(dataset.data)

# 输出数据集的目标向量
print("目标向量:")
print(dataset.target)

# 输出数据集的类别标签
print("类别标签:")
print(dataset.target_names)
# data：表示数据集的特征数据（即属性），通常是一个二维数组或矩阵，每一行代表一个样本，每一列代表一个特征。
# target：表示数据集的目标变量或类别标签（即种类），用于标识每个样本所属的类别。
X = dataset.data
y = dataset.target

# 完善代码:寻找一个合适的函数按照二八比例划分测试集和数据集数据:
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input, x_test, label, y_test = X_train, x_test, y_train, y_test

# 完善代码:利用pytorch把数据张量化:
input = torch.FloatTensor(input)
# input 变量中的数据将被转换为浮点数类型的 Tensor。
label = torch.LongTensor(label)
# label 变量中的数据将被转换为长整数类型的 Tensor。
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)
# 对应转化

label_size = int(np.array(label.size()))


# 将label的大小（维度）转换为一个整数值

# 搭建专属于你的神经网络 它有着两个隐藏层,一个输出层
# 请利用之前所学的知识,填写各层输入输出参数以及激活函数.
# 两个隐藏层均使用线性模型和relu激活函数 输出层使用softmax函数(dim参数设为1)(在下一行注释中写出softmax函数的作用哦)

class NET(nn.Module):
    # 定义一个名为NET的类并继承
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(NET, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)  # 线性层
        self.relu1 = nn.ReLU()  # 创建了一个 ReLU 激活函数层

        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)  # 线性层
        self.relu2 = nn.ReLU()

        self.out = nn.Linear(n_hidden2, n_output)
        self.softmax = nn.Softmax(dim=1)
        # dim=1 表示对输入张量的第一个维度进行 Softmax 运算。
        # 将模型的输出转换为概率分布,使得输出值在 [0, 1] 范围内，并且总和为 1。

    # 前向传播函数
    def forward(self, x):
        hidden1 = self.hidden1(x)  # 输入经过第一个全连接层
        relu1 = self.relu1(hidden1)  # 应用激活函数

        # 完善代码:
        hidden2 = self.hidden2(relu1)  # 输入经过第二个全连接层
        relu2 = self.relu2(hidden2)  # 应用激活函数

        out = relu2  # 输出结果

        return out

    # 测试函数
    def test(self, x):
        y_pred = self.forward(x)  # 将输入数据 x 传递给网络模型进行前向传播
        y_predict = self.softmax(y_pred)  # 将模型的输出 y_pred 通过 softmax 层进行转换，得到预测的类别概率分布 y_predict

        return y_predict


# 定义网络结构以及损失函数
# 完善代码:根据这个数据集的特点合理补充参数,可设置第二个隐藏层输入输出的特征数均为20
# 由数据集特点，特征数=4，类别数=3，
net = NET(n_feature=4, n_hidden1=20, n_hidden2=20, n_output=3)
# 选一个你喜欢的优化器
# 举个例子 SGD优化器 optimizer = torch.optim.SGD(net.parameters(),lr = 0.02)
# 完善代码:我们替你选择了adam优化器,请补充一行代码
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
# lr=0.02 是指定了学习率（learning rate）为 0.02

# 这是一个交叉熵损失函数,不懂它没关系(^_^)
# 交叉熵是一种常见的分类损失函数，尤其适用于二分类或多分类问题。它基于预测概率与真实标签之间的差异进行计算。
loss_func = torch.nn.CrossEntropyLoss()
costs = []  # 存储每个迭代步的损失值
# 完善代码:请设置一个训练次数的变量(这个神经网络需要训练2000次)
epochs = 2000

# 训练网络
# 完善代码:把参数补充完整
for epoch in range(epochs):
    cost = 0
    # 完善代码:利用forward和损失函数获得out(输出)和loss(损失)
    out = net.forward(input)
    loss = loss_func(out, label)
    # 请在下一行注释中回答zero_grad这一行的作用
    # 清除梯度，用于准备进行新一轮的梯度更新，以免得到w1+w2+...的结果。
    optimizer.zero_grad()
    # 完善代码:反向传播 并更新所有参数
    loss.backward()
    optimizer.step()
    cost = cost + loss.cpu().detach().numpy()
    costs.append(cost / label_size)  # 计算平均损失值
# 可视化
plt.plot(costs)
plt.show()

# 测试训练集准确率
out = net.test(input)
prediction = torch.max(out, 1)[1]
pred_y = prediction.numpy()
target_y = label.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("训练集准确率为", accuracy * 100, "%")

# 测试测试集准确率
out1 = net.test(x_test)
prediction1 = torch.max(out1, 1)[1]
pred_y1 = prediction1.numpy()
target_y1 = y_test.numpy()

accuracy1 = float((pred_y1 == target_y1).astype(int).sum()) / float(target_y1.size)
print("测试集准确率为", accuracy1 * 100, "%")

# 至此,你已经拥有了一个简易的神经网络,运行一下试试看吧
# 最后,回答几个简单的问题,本次的问题属于监督学习还是无监督学习呢?batch size又是多大呢?像本题这样的batch size是否适用于大数据集呢,原因是?
# 基于标签，监督学习。
# batch size是整个数据集。对大数据集不可行，会出现内存不足的问题，应该采用的是mini-batch。