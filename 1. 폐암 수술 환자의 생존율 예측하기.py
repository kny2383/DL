import numpy as np
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense
Data_set = np.loadtxt("./data/ToraricSurgery.csv", delimiter=",") # delimiter : 구분값

#1. 데이터 전처리
X = Data_set[:,0:17] # 2차원 넘파이 배열
y = Data_set[:,17] # 1차원 넘파이 배열

#2. 모델 표현방법(딥러닝: 인공신경망)
# clf = svm.SVC()
# clf.fit(X,y)
model = Sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#딥러닝 실행
#모델을 학습하기 위한 환경설정
model.compile(loss='binary_crossentropy', # loss: 오차함수
                optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X, y, epochs=5, batch_size = 16) # epochs : 반복횟수, batch_size : 샘플 수
