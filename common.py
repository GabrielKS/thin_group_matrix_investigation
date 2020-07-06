import numpy as np
import random
import math
import numba

dtype = np.int64  # NumPy datatype
numba_dtype = numba.int64  # Numby datatype
numba_dtype_str = "int64"
hashtype = np.int64  # I hoped that a smaller datatype for hashes might make comparisons faster. It doesn't seem to help very much. TODO: figure out why reduction_investigation is no longer acception int16
numba_hashtype = numba.int64
warning_threshold = 2**53  # Warn us if numbers get this big

A = np.array([[1,1,2],[0,1,1],[0,-3,-2]], dtype=dtype)
B = np.array([[-2,0,-1],[-5,1,-1],[3,0,1]], dtype=dtype)
I3 = np.identity(3, dtype=dtype)
O3 = np.zeros((3,3), dtype=dtype)
O3_1 = np.zeros((1,3,3), dtype=dtype)
O1 = np.zeros((1), dtype=dtype)

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

def count_true(arr):  # Massage the boolean array into a form Numba can handle
    # if len(arr) == 0: return 0  # This line solves the error: "TypeError: No matching definition for argument type(s) array(float64, 1d, C)" and does not appear to be necessary if type is not specified for count_true_numba.
    return count_true_numba(np.array(arr))  # Numba doesn't like dealing with non-NumPy arrays (TODO: solve this problem better)

@numba.jit(nopython=True)
def count_true_numba(arr):  # Count the number of True values in a boolean array. Substitute for return np.count_nonzero(arr)
    result = 0
    for b in arr:
        if b: result += 1
    return result

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
def warn_3x3(a):
    return abs(a[0,0]) >= warning_threshold or abs(a[0,1]) >= warning_threshold or abs(a[0,2]) >= warning_threshold or abs(a[1,0]) >= warning_threshold or abs(a[1,1]) >= warning_threshold or abs(a[1,2]) >= warning_threshold or abs(a[2,0]) >= warning_threshold or abs(a[2,1]) >= warning_threshold or abs(a[2,2]) >= warning_threshold

@numba.jit(nopython=True)
def equals_lazy_3x3(a, b):  # Equivalent to np.equal(a, b).all()
    return a[0,0]==b[0,0] and a[0,1]==b[0,1] and a[0,2]==b[0,2] and a[1,0]==b[1,0] and a[1,1]==b[1,1] and a[1,2]==b[1,2] and a[2,0]==b[2,0] and a[2,1]==b[2,1] and a[2,2]==b[2,2]

@numba.jit(numba_hashtype(numba.types.Array(numba_dtype, 2, "A", readonly=True)), nopython=True)
def hash_3x3(mat):  # A substitute for hash(mat.tobytes())
    return hash((mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2]))

@numba.jit(nopython=True)
def multiply_ref_A(ref):
    return ref << 1

@numba.jit(nopython=True)
def multiply_ref_B(ref):
    return ref << 1 | 1

@numba.jit(nopython=True)
def list_with_element(elem):
    l = numba.typed.List.empty_list(numba_dtype)
    l.append(elem)
    return l

@numba.jit(nopython=True)
def print_program_start():  # Print something conspicuous so it's easy to see the beginning of a program's output when scrolling up through lots of text in a terminal
    print(("="*100+"\n")*10)