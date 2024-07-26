# 2.3.1 标量
import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x+y, x*y, x/y, x**y)

# 2.3.2 向量
x = torch.arange(4)
"""
我们可以使用下标来引用向量的任一元素，例如可以通过
来引用第
个元素。 注意，元素
是一个标量，所以我们在引用它时不会加粗。 大量文献认为列向量是向量的默认方向，在本书中也是如此。
"""
print(x[3])
print(len(x))
print(x.shape)

# 2.3.3 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
# 注意这里是按照位来比较
print(B == B.T)

# 2.3.4 张量
X = torch.arange(24).reshape(2,3,-1)
print(X)

# 2.3.5 张量算法
A = torch.arange(20, dtype=torch.float32).reshape(5,-1)
B = A.clone()
print(id(A) == id(B))
print(A, A + B)

"""
具体而言，两个矩阵的按元素乘法称为Hadamard积（Hadamard product）
"""
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, -1)
print(a + X, (a * X).shape)

# 2.3.6 降维

"""
我们可以对任意张量进行的一个有用的操作是计算其元素的和。 数学表示法使用sigma符号表示求和
"""

x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())

print(A.shape, A.sum())

"""
默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。 我们还可以指定张量沿哪一个轴来通过求和降低维度。 以矩阵为例，为了通过求和所有行的元素来降维（轴0），可以在调用函数时指定axis=0。 由于输入矩阵沿0轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失
"""

A_sum_axis0 = A.sum(axis = 0)
A_sum_axis1 = A.sum(axis = 1)
print(A_sum_axis0, A_sum_axis1, A_sum_axis0.shape, A_sum_axis1.shape)

"""
沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和。
"""

A_sum_axis01 = A.sum(axis=[0,1])
print(A_sum_axis01)

"""
一个与求和相关的量是平均值（mean或average）。 我们通过将总和除以元素总数来计算平均值。 在代码中，我们可以调用函数来计算任意形状张量的平均值。
"""

print(A.mean(), A.sum()/A.numel())

"""
同样，计算平均值的函数也可以沿指定轴降低张量的维度。
"""

print(A.mean(axis = 0), A.sum(axis = 0) / A.shape[0])

# 2.3.6.1. 非降维求和

"""
但是，有时在调用函数来计算总和或均值时保持轴数不变会很有用。
"""
sum_A = A.sum(axis = 1, keepdims = True)
print(sum_A)

""""
例如，由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A
"""

print(A / sum_A)

"""
如果我们想沿某个轴计算A元素的累积总和， 比如axis=0（按行计算），可以调用cumsum函数。 此函数不会沿任何轴降低输入张量的维度。
"""
print(A)
print(A.cumsum(axis=0))

# 2.3.7. 点积（Dot Product）
y = torch.ones(4, dtype=torch.float32)
print(x)
print(x, y, torch.dot(x, y))

""""
在代码中使用张量表示矩阵-向量积，我们使用mv函数。 当我们为矩阵A和向量x调用torch.mv(A, x)时，会执行矩阵-向量积。 注意，A的列维数（沿轴1的长度）必须与x的维数（其长度）相同。
"""
print("-----------------------------------------------------------------------------------")
print(A)
print(x)
print(A.shape, x.shape, torch.mv(A, x))

# 2.3.9. 矩阵-矩阵乘法

B = torch.ones(4, 3)
print(B)
print(A)
print(torch.mm(A, B))

# 2.3.10 范数(norm)

"""
线性代数中最有用的一些运算符是范数（norm）。 非正式地说，向量的范数是表示一个向量有多大。 这里考虑的大小（size）概念不涉及维度，而是分量的大小。
"""

## L2范数为欧几里得距离
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

## L1范数为绝对值相加总和

print(torch.abs(u).sum())










