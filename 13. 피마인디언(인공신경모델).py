import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 읽기
df = pd.read_csv("./data/pima-indians-diabetes.csv")
X = df.iloc[:,0:8] # 세부 정보를 X로 지정, train data
y = df.iloc[:,8] # 당뇨병 여부를 y로 지정, 정답 레이블

#2. 딥러닝 모델 생성
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu', name = 'Dense_1')) # name : 노드의 이름
model.add(Dense(8, activation = 'relu', name = 'Dense_2'))
model.add(Dense(1,activation = 'sigmoid', name = 'Dense_3'))
model.summary() # 층과 층의 연결을 한눈에 볼 수 있게 해줌

#모델을 컴파일 환경 설정
#앞서 지정한 모델이 효과적으로 구현될 수 있게 여러 가지 환경을 설정해 주면서 컴파일 하는 부분
#손실 함수는 최적의 가중치를 학습하기 위해 필수적인 부분이다.
#올바른 손실 함수를 통해 계선된 오차는 옵티마이저를 적절히 활용하도록 만들어준다.
model.compile(loss = 'binary_crossentropy', 
                optimizer = 'adam', metrics = ['accuracy']) # metircs = ['accuracy'] : 학습셋에 대한 정확도에 기반해 결과를 출력한다.

#모델을 기계학습
model.fit(X, y, epochs = 100, batch_size = 5) 
# epochs = 100 : 각 샘플이 처으부터 끝까지 백 번 재사용될 때까지 실행을 반복하라는 의미
# batch_size: 샘플을 한 번에 몇 개씩 처리할지 정하는 부분
print("정확도: %.4f" %(model.evaluate(X, y)[1]))
