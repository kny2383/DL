import numpy as np
import matplotlib.pyplot as plt

# 딥러닝 인공신경망 생성 클래스
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([2,4,6,8]) 
y = np.array([81,93,91,97])

# 모델 생성
model = Sequential()

model.add(Dense(1, input_dim = 1, activation = 'linear'))
model.compile(optimizer = 'sgd', loss = 'mse')
model.fit(x, y, epochs = 2100)

plt.scatter(x,y)
plt.scatter(x, model.predict(x))
plt.plot(x, model.predict(x), 'r')
plt.show()

# 임의의 시간을 넣어 점수를 예측
hour = 7
pre = model.predict([hour])
print("%.f시간을 공부할 경우 예상 점수: %.02f점" %(hour,pre))