import numpy as np
import random

dtype = np.int
A = np.matrix([[1,1,2],[0,1,1],[0,-3,-2]], dtype=dtype)
B = np.matrix([[-2,0,-1],[-5,1,-1],[3,0,1]], dtype=dtype)
I3 = np.identity(3, dtype=dtype)

def random_H_str(length):
    result = ""
    for i in range(length):
        r = random.random() > 0.5
        m = "A" if r else "B"
        result += m
    return result

def random_H(length):
    import conversion
    return conversion.h_to_mat(random_H_str(length))