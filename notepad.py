from common import *
from conversion import *
import numpy as np
import numba

# full_str = "BBABB"
# h1_str = "BBABBAABAA"
# h2_str = "BBABBABBAB"
# full_ref = 100  # 1938285, 1 1101100100 1101101101
# h1_ref = 1892  # 1 1101100100
# h2_ref = 1901  # 1 1101101101
# str_length = ref_to_length(full_ref)
# cache_length = 10
# n = min(cache_length, str_length)

# # print(int_to_h(full_ref))
# # print(max(full_ref >> cache_length, 1))
# # print((full_ref & ((1 << n)-1) | (1 << n)))
# # print(ref_last_n(full_ref, cache_length))
# print(full_ref)
# print(bin(full_ref))
# print(bin(ref_last_n(full_ref, cache_length)))
# print(ref_first_n(full_ref, cache_length))
# print(ref_last_n(full_ref, cache_length))

# print(np.argmax([False, True]))

# for crr in (19173961, 26364196, 38347922, 52728393, 76695844, 105456787):
#     print(crr)
#     print(ref_to_length(crr))
#     print(int_to_h(crr))
#     print(h_to_mat(int_to_h(crr)))
#     print()

# X = h_to_mat("BAABAABAABAABAABAABAABAA")
# print(A*A*X)
# print(X*B*B)
# print(B*A*A)

# print(ref_to_length(1947355))

# @numba.jit(numba_dtype(numba_dtype[:,:], numba_dtype[:,:,:]), nopython=True)  # In this case, if we don't manually specify type, Numba actually slows things down.
# def find_first_equal(elem, elem_list):  # Returns the index of the first value of elem_list equal to elem, or -1 if there is no match
#     return 0
#     # for i in range((len(elem_list))):
#     #     if np.equal(elem, elem_list[i,:,:]).all(): return i
#     # return -1

# ffe = find_first_equal(np.zeros((3,3), dtype=dtype), np.zeros((3,3,3), dtype=dtype))-1
# print(ffe)

# print(ref_to_length(663336102817))

# print(ref_to_h(multiply_ref_A(h_to_ref("AABBAB"))))
# print(ref_to_h(multiply_ref_B(h_to_ref("AABBAB"))))

# l = numba.typed.List()
# l.append(2)
# print(repr(l))
# print(numba.types.ListType)
# print(numba.types.int64)
# print(numba.types.ListType(numba.types.int64))
# l = numba.typed.List.empty_list(numba.types.int64)
# l.append(2)
# l.append(2.2)  # Errors

# print(type(hash_3x3(A)))

print(ref_to_h(14380199))
print(h_to_ref("BBB"))