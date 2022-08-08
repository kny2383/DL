import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEnconder
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('./data/iris3.csv', 
                names=["sepal_length","sepal_width","petal_legth","species"])
# 전체 상관도를 볼 수 있는 그래프 출력
sns.pairplot(df, hue = 'species')
plt.show()

dataset = df.values 

X = dataset[:0,4].astype(float)
y_obj = dataset[:,4]

e = LabelEnconder() # 문자열 -> 숫자 변환
e.fit(y_obj)
y = e.transform(y_obj)

# one - hot encoding 
# 다 0인데 하나만 hot하다.
# => 다중분류문제에서 정답레이블을 반드시 one - hot encoding

y_encode = tf.keras.utils.to_categorical(y) # one - hot encoding

# 인공신경망 모델 생성
model = Sequential()
model.add(Dense(12, input_dim = 4, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.summary()

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy', # categorical_crossentropy: 범주형 교차 엔트로피, 여럿 중 하나를 예측할 때 사용하는 오차함수
                opimizer = 'adam', metrics = ['accuracy'])

# 모델 실행
history = model.fit(X, y_encode, epochs = 50, batch_size = 5)

# 정확도와 손실함수를 리턴하는 evaluate() 함수
result = model.evaluate(X,y_encode)
print(result[1])