import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # 3차원 그래프

x1 = np.array([2,4,6,8]) # 공부시간
x2 = np.array([0,4,2,3]) # 과외 횟수
y = np.array([81,93,91,97]) # 성적

plt.rc('font', family="Malgun Gothic")
ax = plt.axes(projection = '3d') # 그래프 유형 설정
ax.set_xlabel('공부시간')
ax.set_ylabel('과외 횟수')
ax.set_zlabel('성적')
ax.scatter(x1,x2,y)
plt.show()