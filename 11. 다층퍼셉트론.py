import numpy as np

# [문제1] 앞의 AND, OR, NAND 게이트를 이용하여 XOR 문제를 해결하는 코드

def AND(x):
    w = np.array([0.5,0.5])
    b = -0.7

    # 가중합
    ws = (x * w).sum() + b

    if ws >= 0:
        return 1
        
    return 0 

def OR(x):
    w = np.array([0.5,0.5])
    b = -0.2

    # 가중합
    ws = (x * w).sum() + b

    if ws >= 0:
        return 1
        
    return 0 

def NAND(x):
    w = np.array([-0.5,-0.5])
    b = 0.7

    # 가중합
    ws = (x * w).sum() + b

    if ws >= 0:
        return 1
        
    return 0 

x_data = np.array([[0,0],[0,1],[1,0],[1,1]])

for x in x_data:
    print("AND(%d, %d) = %d" %(x[0], x[1], AND(x)))
for x in x_data:
    print("OR(%d, %d) = %d" %(x[0], x[1], OR(x)))
for x in x_data:
    print("NAND(%d, %d) = %d" %(x[0], x[1], NAND(x)))

for x in x_data:
    n1 = NAND(x)
    n2 = OR(x)
    s = np.array([n1,n2])
    y = AND(s)
    print("x1: %d, x2: %d, n1: %d, n2: %d, y: %d" %(x[0],x[1],n1,n2,y))

# 위의 각 함수를 통해서 다층 퍼셉트론처럼 만들었지만 다층 퍼셉트론은 아니다
# 단층 퍼셉트론 3개를 이용해서 XOR 연산을 수행함

# [문제2] AND, OR, NAND함수를 통합하는 GATE 함수를 생성하고 다음 문제를 해결하는 코드

def GATE(x,w,b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1

w1 = np.array([[-0.5,-0.5],[0.5,0.5]]) #NAND #OR
b1 = np.array([0.7,-0.2])
w2 = np.array([0.5,0.5]) #AND
b2 = np.array([-0.7])

for x in x_data:
    n1 = GATE(x,w1[0],b1[0]) # 은닉층 : NAND
    n2 = GATE(x,w1[1],b1[1]) # 은닉층 : OR

    s = np.array([n1,n2]) 
    y = GATE(s,w2,b2[0]) # 출력층

    print("x1: %d, x2: %d, n1: %d, n2: %d, y: %d" %(x[0],x[1],n1,n2,y))

