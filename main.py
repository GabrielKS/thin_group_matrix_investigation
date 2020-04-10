import numpy as np
import random
import permutations

dtype = np.int
A = np.matrix([[1,1,2],[0,1,1],[0,-3,-2]], dtype=dtype)
B = np.matrix([[-2,0,-1],[-5,1,-1],[3,0,1]], dtype=dtype)
I3 = np.identity(3, dtype=dtype)

def main():

    # print(A)
    # print(B)

    # print(A*A*A)
    # print(B*B*B)

    # for i in range(10000):
    #     s = random_H_str(8)
    #     m = h_str_to_mat(s)
    #     if np.array_equal(m, I3): print(s)

    # permutations.main()
    # print((A*B*A*A*B*B)**178)
    # print(A*A)
    print(A*A)

def run_test(fn, times):
    passes = True
    for i in range(times):
        m = random_H(10)
        if not fn(m): passes = False
    print(passes)

def is_001(m):
    return m[2,0]%3 == 0 and m[2,1]%3 == 0 and m[2,2]%3 == 1

def is_01_implies_02(m):
    if m[0,1]%3 == 0:
        if m[0,2]%3 == 0: return True
        return False
    return True

def random_H_str(length):
    result = ""
    for i in range(length):
        r = random.random() > 0.5
        m = "A" if r else "B"
        result += m
    return result

def h_str_to_mat(s):
    result = I3
    for i in range(len(s)):
        m = A if s[i] == "A" else B
        result = result*m
    return result

def random_H(length):
    return h_str_to_mat(random_H_str(length))

if __name__ == "__main__":
    main()
