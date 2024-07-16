"""
2. ⽤其他形状（例如三维张量）替换⼴播机制中按元素操作的两个张量。结果是否与预期相同？
"""
import torch
"""
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
"""
X = torch.arange(24).reshape((2, 3, 4))
Y = torch.tensor([ [0,1,0], [1,0,1], [2, 3, 2] ])
print(X)
print(Y)
print(X + Y)

A = torch.arange(24).reshape((2, 3, 4))
print("A:")
print(A)

# 创建一个形状为 (1, 3, 1) 的三维张量 B
B = torch.tensor([[[1], [2], [3]]])
print("\nB:")
print(B)

