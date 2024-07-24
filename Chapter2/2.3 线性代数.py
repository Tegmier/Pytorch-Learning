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