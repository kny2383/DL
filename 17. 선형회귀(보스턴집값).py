import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

df = pd.reade_csv("./data/house_train.csv", header = None, delim_withespace = True)

dataset = df.values
X = dataset[:,0:13]
y = dataset[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

model = Sequential()
model.add(Dense(30, input_dim = 13, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1)) # 출력층의 활성화 함수가 없다. 가중합만 구한다.
# 가중합이 선형회귀에서는 활성화함수이다.
# 항등함수 : 출력층의 활성화 함수가 없는 것

# 딥러닝 컴파일 환경설정
model.compile(optimizer = 'adam', loss = 'mean_squared_error') # 선형 회귀이므로 평균 제곱 오차

model.fit(X_train, y_train, epochs = 200, batch_size = 32)

# 모델을 이용해서 새로운 data의 예측값을 구한다.
y_pred = model.predict(X_test).flatten()
for i in range(10):
    label = y_test[i]
    pre = y_pred[i]

    print("실제가격:  %.3f, 예상가격: %.3f %(label,pre)")

# 분류 : 정확도
# 회귀 : 근사값