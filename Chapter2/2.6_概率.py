import torch
from torch.distributions import multinomial
from d2l import torch as d2l

"""
在统计学中，我们把从概率分布中抽取样本的过程称为抽样（sampling）。 
笼统来说，可以把分布（distribution）看作对事件的概率分配， 稍后我们将给出的更正式定义。 将概率分配给一些离散选择的分布称为多项分布（multinomial distribution）。
"""
# 定义了一个6类别的多项分布的概率向量
fair_probs = torch.ones([6]) / 6
print(fair_probs)
sampling = multinomial.Multinomial(10, fair_probs).sample()
print(sampling)

# counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# print(counts.shape)
# cum_counts = counts.cumsum(dim=0)
# print(cum_counts)
# estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# d2l.set_figsize((6, 4.5))
# for i in range(6):
#     d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
#     d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
# d2l.plt.gca().set_xlabel('Groups of experiments')
# d2l.plt.gca().set_ylabel('Estimated probability')
# d2l.plt.legend()
# d2l.plt.show()

# exercise

fair_probs = torch.ones([6]) / 6
print(fair_probs)
sampling = multinomial.Multinomial(10, fair_probs).sample((100,))
print(sampling)
cum_sum_sampling = sampling.cumsum(dim = 0)
print(cum_sum_sampling)
total_result = cum_sum_sampling / cum_sum_sampling.sum(dim = 1, keepdim=True)
print(total_result[-1, :])