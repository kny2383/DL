import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/wine.csv', header = None)

X = df.iloc[:,0:12]
y = df.iloc[:,12]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#딥러닝 모델 생성
model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

#딥러닝 컴파일 환경설정
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                metrics=['accuracy'])

#딥러닝 실행
model.fit(X,y,epochs = 50, batch_size = 500, validation_split = 0.25) # 검증데이터를 25%로 적용
model.evaluate(X_test, y_test)[1]

modelpath = './data/model/all/{epoch:02d}-{val_accuracy:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, verbose = 1) # ModelCheckpoint(): 학습 중인 모델을 저장하는 함수, verbose = 1 : True(진행되는 현황 모니터)
model.fit(X,y,epochs = 50, batch_size = 500, validation_split = 0.25, verbose = 0, callbacks  = [checkpointer]) # validation_split : 검증셋