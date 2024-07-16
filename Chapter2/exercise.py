import torch

# arange函数创建一个行向量
x = torch.arange(12)
print(x)
print(x.shape)
print(x.size())

# numel函数返回张量的元素个数
print(x.numel())

# reshape函数改变张量的形状，可以使用-1来自动计算维度
X = x.reshape(3, 4)
print(X)

# 创建全0，全1，其他常量，从特定分布中随机采样的数字来初始化的矩阵，的张量
print(torch.zeros(2, 1, 2, 5))
print(torch.ones(2, 3, 4))
# 高斯采样
print(torch.randn(3,4))
# 基于python列表
print(torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))

# 基本运算符是按照元素来的
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x+y, x-y, x*y, x/y, x**y)
print(torch.exp(x))

# 通过计算维度拼接后的维度大小来判断结果
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim = 0))
print(torch.cat((X, Y), dim = 1))

# 通过逻辑运算构建二元张量
print(X == Y)

# 对所有张量求和
print(sum(X))