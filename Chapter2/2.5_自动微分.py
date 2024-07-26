import torch

x = torch.arange(4.0)
"""
在我们计算
关于
的梯度之前，需要一个地方来存储梯度。 重要的是，我们不会在每次对一个参数求导时都分配新的内存。 因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。 注意，一个标量函数关于向量
的梯度是向量，并且与
具有相同的形状。
"""
x.requires_grad_(True)

print(x.grad)

y = 2 * torch.dot(x, x)

print(y)

y.backward()
print(x.grad)

print(x.grad == 4*x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 2.5.2. 非标量变量的反向传播
x.grad.zero_()
y = x*x
y.sum().backward()
print(x.grad)

# 2.5.3. 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)

print(a.grad == d / a)