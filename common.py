import numpy as np
import random
import math
import numba

dtype = np.int64  # NumPy datatype
numba_dtype = numba.int64  # Numby datatype
A = np.array([[1,1,2],[0,1,1],[0,-3,-2]], dtype=dtype)
B = np.array([[-2,0,-1],[-5,1,-1],[3,0,1]], dtype=dtype)
I3 = np.identity(3, dtype=dtype)
O3 = np.zeros((3,3), dtype=dtype)

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

@numba.jit(nopython=True)
def length_to_ref(length):  # The first reference to a word of length length will be at length_to_ref(length). Also, the number of words of length length will be length_to_ref(length).
    return 2**length

@numba.jit(nopython=True)
def ref_to_length(ref):  # The length of the word represented by ref
    # return int(math.log2(ref))  # Clearer than messing around with bits
    # return ref.bit_length()-1  # Messing around with bits is faster
    result = -1
    while ref > 0:
        result += 1
        ref = ref >> 1
    return result

@numba.jit(nopython=True)
def ref_first_n(ref, n):  # Computes ref corresponding to the word composed of the first ref_to_length(ref)-n letters of ref
    return max(ref >> n, 1)  # We just shift ref to the right n bits. Typically need to do anything about the leading 1 because it just stays there. If n >= ref_to_length(n) and so the leading 1 slides off, we just return 1

@numba.jit(nopython=True)
def ref_last_n(ref, n):  # Computes ref corresponding to the word composed of the last n letters of ref
    n = min(n, ref_to_length(ref))  # If ref is shorter than n letters, we just want all of ref
    return ref & ((1 << n)-1) | (1 << n)  # ANDs with n ones to get the n lowest bits, then ORs with 1 shifted n bits to the left to put the leading 1 back

@numba.jit(nopython=True)
def multiply_3x3(a, b):  # Because Numba doesn't natively support matrix multiplication for integers :(
    result = np.zeros((3,3), dtype)

    result[0,0] = a[0,0]*b[0,0]+a[0,1]*b[1,0]+a[0,2]*b[2,0]
    result[0,1] = a[0,0]*b[0,1]+a[0,1]*b[1,1]+a[0,2]*b[2,1]
    result[0,2] = a[0,0]*b[0,2]+a[0,1]*b[1,2]+a[0,2]*b[2,2]
    
    result[1,0] = a[1,0]*b[0,0]+a[1,1]*b[1,0]+a[1,2]*b[2,0]
    result[1,1] = a[1,0]*b[0,1]+a[1,1]*b[1,1]+a[1,2]*b[2,1]
    result[1,2] = a[1,0]*b[0,2]+a[1,1]*b[1,2]+a[1,2]*b[2,2]

    result[2,0] = a[2,0]*b[0,0]+a[2,1]*b[1,0]+a[2,2]*b[2,0]
    result[2,1] = a[2,0]*b[0,1]+a[2,1]*b[1,1]+a[2,2]*b[2,1]
    result[2,2] = a[2,0]*b[0,2]+a[2,1]*b[1,2]+a[2,2]*b[2,2]

    return result

@numba.jit(nopython=True)
def equals_lazy_3x3(a, b):  # Equivalent to np.equal(a, b).all()
    return a[0,0]==b[0,0] and a[0,1]==b[0,1] and a[0,2]==b[0,2] and a[1,0]==b[1,0] and a[1,1]==b[1,1] and a[1,2]==b[1,2] and a[2,0]==b[2,0] and a[2,1]==b[2,1] and a[2,2]==b[2,2]

@numba.jit(nopython=True)
def mat_hash_3x3(mat):  # A substitute for hash(mat.tobytes())
    return hash((mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2]))
