import numpy as np
import matplotlib.pyplot as plt

X = np.array([2,4,6,8])
Y = np.array([81,93,91,97])

a = 3
b = 76

Y_pred = a * X + b
print("예측값: ", Y_pred)

error = Y - Y_pred
print("오차", error)

#오차의 합
error_sum = (error**2).sum()
print(error_sum)

#평균제곱오차
MSE = error_sum / len(X)
print("평균제곱오차", MSE)

#평균제곱오차 그래프
plt.scatter(X, Y)
plt.scatter(X, Y_pred)
plt.plot(X, Y_pred)
plt.show()