import numpy as np
import matplotlib.pyplot as plt

x = np.arrange(-8,8.1,0.1)
y = 1/(1+np.e**(-x)) #시그모이드 함수의 공식

plt.plot(x,y)
plt.show()