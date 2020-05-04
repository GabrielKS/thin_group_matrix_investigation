from common import *
from conversion import *
import numpy as np

full_str = "BBABB"
h1_str = "BBABBAABAA"
h2_str = "BBABBABBAB"
full_ref = 100  # 1938285, 1 1101100100 1101101101
h1_ref = 1892  # 1 1101100100
h2_ref = 1901  # 1 1101101101
str_length = ref_to_length(full_ref)
cache_length = 10
n = min(cache_length, str_length)

# print(int_to_h(full_ref))
# print(max(full_ref >> cache_length, 1))
# print((full_ref & ((1 << n)-1) | (1 << n)))
# print(ref_last_n(full_ref, cache_length))
print(full_ref)
print(bin(full_ref))
print(bin(ref_last_n(full_ref, cache_length)))
print(ref_first_n(full_ref, cache_length))
print(ref_last_n(full_ref, cache_length))