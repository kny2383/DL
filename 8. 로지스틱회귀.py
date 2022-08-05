import numpy as np
import matplotlib.pyplot as plt

data = [[2,0],[4,0],[6,0],[8,1],[10,1],[12,1],[14,1]]

x_data = [x_y[0] for x_y in data]
y_data = [x_y[1] for x_y in data]

plt.scatter(x_data, y_data)
plt.show()

a = 0
b = 0
lr = 0.05
epochs = 2001

# 시그모이드 함수 정의
def sigmoid(x):
    return 1/(1+np.e**(-x))

# 경사하강법 실행
for i in range(epochs):
    for x, y in data:
        
        # 예측값 : sigmoid(a * x + b)
        a_diff = x * (sigmoid(a * x + b)-y) # 오차
        b_diff = sigmoid(a * x + b) - y # -오차

        # 기울기 a와 절편 b를 업데이트
        a = a - lr * a_diff
        b = b - lr * b_diff

        if i % 100 == 0:
            print("epoch = %4d, a: %.2f, b: %.2f" %(i,a,b))

y_pred =[sigmoid(a * x + b) for x in x_data]
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred)
plt.show() 