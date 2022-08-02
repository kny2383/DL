import numpy as np
import matplotlib.pyplot as plt

# X : 공부시간
# Y: 성적

data = [[2,81],[4,93],[6,91],[8,97]]
X = [i[0] for i in data] # 리스트 컨프레이션
Y = [i[1] for i in data]

plt.scatter(X,Y)
plt.show()

x = np.array(X)
y = np.array(Y)

a = 0 # 기울기
b = 0 # 절편

lr = 0.03 # 학습률

#몇 번 반복될지 설정
epochs = 2001

n = len(x)

for i in range(epochs):
    y_pred = a * x + b
    error = y - y_pred

    a_diff = (2/n) * sum(-x * (error))
    b_diff = (2/n) * sum(-(error))

    a = a - lr * a_diff
    b = b - lr * b_diff
    
    if i % 100 == 0:
        print("epoch= %.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))

y_pred = a * x + b
plt.scatter(x,y)
plt.scatter(y_pred)
plt.plot(x, y_pred) # 예측선
plt.show()