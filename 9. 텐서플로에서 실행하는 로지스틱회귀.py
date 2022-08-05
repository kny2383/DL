import numpy as np
import matplotlib.pyplot as plt

# 딥러닝 인공신경망 생성 클래스
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([2,4,6,8,10,12,14])
y = np.array([0,0,0,1,1,1,1])

# 딥러닝: 인공신경망 모델 생성
model = Sequential() # 모델생성하는 클래스
model.add(Dense(1, input_dim = 1, activation = 'sigmoid'))
model.compile(optimizer = 'sgd', loss = 'binaray_crossentropy')
model.fit(x,y,epochs=5000)

plt.scatter(x,y)
plt.plot(x, model.predict(x),'r')
plt.show()

# 임의의 학습시간의 합격 예상 확률
hour = 7
pred = model.predict([hour])
print("7시간의 합격 예상 확률: ", pred)