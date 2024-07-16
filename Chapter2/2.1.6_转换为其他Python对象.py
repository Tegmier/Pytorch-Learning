import torch

"""
将深度学习框架定义的张量转换为NumPy张量（ndarray）很容易，反之也同样容易。 torch张量和numpy数
组将共享它们的底层内存，就地操作更改⼀个张量也会同时更改另⼀个张量。
"""

X = torch.arange(12, dtype=torch.float32).reshape((3,4))

A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

"""
要将⼤⼩为1的张量转换为Python标量，我们可以调⽤item函数或Python的内置函数。
"""

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))