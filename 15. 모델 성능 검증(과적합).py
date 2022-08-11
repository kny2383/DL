import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

#1. 데이터 읽기
df = pd.DataFrame('./data/sonar.csv', header = None)

X = dataset[:,:60]
X.astype(np.float64)
y_obj = dataset[:,60]
e = LabelEncoder()
e.fit(y_obj)

y = e.transform(y_obj)

#딥러닝 모델 생성
model = Sequential()
model.add(Dense(24, input_dim = 60, activaiton = 'relu'))
model.add(Dense(10, activaiton = 'relu'))
model.add(Dense(1, activaiton = 'sigmoid'))

#딥러닝 컴파일 환경설정
model.compile(loss = 'binary_croseentropy',
                optimizer = 'adam', metrics = ['accuracy'])

#딥러닝 실행
model.fit(X, y, epochs = 200, batch_size = 10)
model.evaluate(X,y)[1]

#################################################
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

#새 모델 생성
model = Sequential()
model.add(Dense(24, input_dim = 60, activaiton = 'relu'))
model.add(Dense(10, activaiton = 'relu'))
model.add(Dense(1, activaiton = 'sigmoid'))
model.compile(loss = 'binary_croseentropy',
                optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 200, batch_size = 10)
model.evaluate(X_test,y_test)[1] 

#학습데이터의 정확도는 100프로이고, 테스트데이터의 정확도는 81프로이다.

model.save('my_model.hdf5')
del model

new_model = load_model('my_model.hdf5') # model 읽어오기
