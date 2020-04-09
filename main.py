import numpy as np
import random

A = np.matrix([[1,1,2],[0,1,1],[0,-3,-2]])
B = np.matrix([[-2,0,-1],[-5,1,-1],[3,0,1]])

def main():

    # print(A)
    # print(B)

    # print(A*A*A)
    # print(B*B*B)

    passes = True
    for i in range(1000):
        m = random_H(10)
        if not is_01_implies_02(m): passes = False
    print(passes)

def is_001(m):
    return m[2,0]%3 == 0 and m[2,1]%3 == 0 and m[2,2]%3 == 1

def is_01_implies_02(m):
    if m[0,1]%3 == 0:
        if m[0,2]%3 == 0: return True
        return False
    return True

def random_H(length):
    result = np.identity(3, dtype=int)
    for i in range(length):
        r = random.random() > 0.5
        m = A if r else B
        result = result*m
    return result

if __name__ == "__main__":
    main()
    