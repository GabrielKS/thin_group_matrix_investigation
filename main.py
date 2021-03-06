from common import *
import random
from conversion import h_to_mat
import permutations

#testing comment testing

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
    #print(A*A)

    # run_test(mod10_11,1000)
    # run_test(is_noneven, 10000)
    pass

def run_test(fn, times):
    passes = True
    for i in range(times):
        m = random_H(20)
        if not fn(m): passes = False
    print(passes)

def is_noneven(m):
    return m[0,0]%2!=0 or m[0,1]%2!=0 or m[0,2]%2!=0 or m[1,0]%2!=0 or m[1,1]%2!=0 or m[1,2]%2!=0

def is_001(m):
    return m[2,0]%3 == 0 and m[2,1]%3 == 0 and m[2,2]%3 == 1

def mod00_01(m):
    return not (m[0,0]%3 == 0 and m[0,1]%3 == 0)

def mod00_10(m):
    return not (m[0,0]%3 == 0 and m[1,0]%3 == 0)

def mod10_11(m):
    return not (m[1,0]%3 == 0 and m[1,1]%3 == 0)

def mod01_11(m):
    return not (m[0,1]%3 == 0 and m[1,1]%3 == 0)

def is_01_implies_02(m):
    if m[0,1]%3 == 0:
        if m[0,2]%3 == 0: return True
        return False
    return True

if __name__ == "__main__":
    main()
