import numpy as np

X = [[0,0],[0,1],[1,0],[1,1]]

w1 = 0.5
w2 = 0.5
theta = 0.7 # 임계값

# 가중합 >= 임계값 True or False
print(w1 * 0 + w2 * 0 >= theta)
print(w1 * 1 + w2 * 0 >= theta)
print(w1 * 0 + w2 * 1 >= theta)
print(w1 * 1 + w2 * 1 >= theta)

for x1, x2 in X:
    print("x1: ", x1, "x2: ", x2, w1 * x1 + w2 * x2)

def AND(x1, x2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.7 # 임계값
    
    # 가중합
    ws = w1 * x1 + w2 * x2
    
    if ws >= theta:
        return 1

    return 0

# 단층 퍼셉트론
for x1, x2 in X:
    print('AND(%d, %d) = %d' %(x1,x2,AND(x1,x2)))

# 위 코드는 퍼셉트론이지만 완성된 가중합은 아니다.
# 임계값 theta가 계산식 안에 들어오면 이것이 바로 편향(bias)
# 따라서 편향은 -theta
# (w1 * x1 + w2 * x2 - theta) >= 0 -> True

# 완성된 가중합 코드
#def AND(x1, x2):
#    w1 = 0.5
#    w2 = 0.5
#    b = -0.7 # 임계값
    
    # 가중합
#    ws = w1 * x1 + w2 * x2 + b
    
#    if ws >= 0:
#        return 1

#    return 0

def OR(x1,x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.2

    ws = w1 * x1 + w2 * x2 + b

    if ws >= 0:
        return 1

    return 0

for x1, x2 in X:
    print('OR(%d, %d) = %d' %(x1,x2,OR(x1,x2)))

def NAND(x1, x2):
    w1 = -0.5
    w2 = -0.5
    theta = 0.7 # 임계값
    
    # 가중합
    ws = w1 * x1 + w2 * x2
    
    if ws >= 0:
        return 1

    return 0

for x1, x2 in X:
    print('NAND(%d, %d) = %d' %(x1,x2,NAND(x1,x2)))