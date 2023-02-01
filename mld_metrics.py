import numpy as np
import math

from sklearn.datasets import make_multilabel_classification


# imbalance rate per label
def ir_per_label(label, y):
    a = np.sum(y[:, label], axis=0) # 这个类别 有多少个样本
    b = np.max(np.sum(y, axis=0))  # 所有类别下面 最多的样本数量

    return b / a, a


# mean imbalance rate
def mean_ir(y):
    mean = 0.0
    ir_list = []
    for i in range(y.shape[1]): # 所有可能的 样本类别
        ir, a =  ir_per_label(i, y)
        if a != 0:
            mean += ir
        ir_list.append(ir_per_label(i, y))
    #   最多类别数量 / 每个类别数量
    # / 样本数量
    tmp = ir_list[299:]
    return mean / y.shape[1]


# used to calculate cvir
def ir_per_label_alpha(y, mean_ir_val):
    mean = 0.0

    for i in range(y.shape[1]):
        ir, a = ir_per_label(i, y)
        if a != 0:
            mean += ((ir - mean_ir_val) ** 2) / (y.shape[1] - 1)

    return math.sqrt(mean)


# coefficient of variation of IRperLabel
def cvir(y):
    mean_ir_val = mean_ir(y)
    alpha = ir_per_label_alpha(y, mean_ir_val)

    return alpha / mean_ir_val


'''
# Example of usage
x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8)

print('Positive samples per class:')
print(np.sum(y, axis=0))

irs = []

for i in range(y.shape[1]):
    irs.append(ir_per_label(i, y))

print('Imbalance Rate per class:')
print(irs)

print('Mean Imbalance Rate: ')
print(mean_ir(y))

print('Coefficient of variation per label:')
print(cvir(y))
'''
