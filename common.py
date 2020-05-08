import numpy as np
import random
import math

dtype = np.int64
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

def count_true(arr):  # Count the number of True values in a boolean array
    return np.count_nonzero(arr)

def length_to_ref(length):  # The first reference to a word of length length will be at length_to_ref(length). Also, the number of words of length length will be length_to_ref(length).
    return 2**length

def ref_to_length(ref):  # The length of the word represented by ref
    # return int(math.log2(ref))  # Clearer than messing around with bits
    return ref.bit_length()-1  # Messing around with bits is faster

def ref_first_n(ref, n):  # Computes ref corresponding to the word composed of the first ref_to_length(ref)-n letters of ref
    return max(ref >> n, 1)  # We just shift ref to the right n bits. Typically need to do anything about the leading 1 because it just stays there. If n >= ref_to_length(n) and so the leading 1 slides off, we just return 1

def ref_last_n(ref, n):  # Computes ref corresponding to the word composed of the last n letters of ref
    n = min(n, ref_to_length(ref))  # If ref is shorter than n letters, we just want all of ref
    return ref & ((1 << n)-1) | (1 << n)  # ANDs with n ones to get the n lowest bits, then ORs with 1 shifted n bits to the left to put the leading 1 back