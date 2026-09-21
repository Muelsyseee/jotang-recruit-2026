import numpy as np

scores = {
    "张三": 90,
    "李四": 80,
    "王五": 70
}

def average(scores):
    return  sum(scores.values())/len(scores)

def highest(scores):
    return max(scores.items(), key=lambda item: item[1])

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

C = A @ B

def output():
    print("矩阵乘法结果：", C)
    print("A shape：", A.shape)
    print("B shape：", B.shape)
    print("C shape：", C.shape)


output()