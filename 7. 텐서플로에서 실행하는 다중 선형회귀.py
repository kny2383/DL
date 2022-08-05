import numpy as np
import matplotlib.pyplot as plt

# 딥러닝 인공신경망 생성 클래스
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[2,0],[4,4],[6,2],[8,3]])
y = np.array([81,93,91,97])

# 딥러닝 모델 생성
model = Sequential()
model.add(Dense(1, input_dim = 2, activation = 'linear'))
model.compile(optimizer = 'sgd', loss = 'mse')
model.fit(x, y, epochs = 2000)

# 임의의 학습시간과 과외횟수로 점수를 예측하는 딥러닝 모델 만들기
hour = 7
private_class = 4
pre = model.predict([hour, private_class])
print("예상 점수: ", pre)