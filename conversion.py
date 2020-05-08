from common import *


def main():
    #print(int_to_bin(21))
    print(int_to_h(76695844))
    print(h_to_mat(int_to_h(76695844)))
    #print(h_to_mat(int_to_h(21)))
    print("done")

    '''
    # Conversion between numbers and binary strings
    print(bin_to_int("1010011"))
    print(int_to_bin(83))
    print()

    # Conversion between binary strings and unreduced h (here, h is in reduced form)
    print(bin_to_h_unreduced("0000000001010011"))
    print(h_to_bin_unreduced("ABAABB"))
    print()

    # Conversion between binary strings and unreduced h (here, h is not in reduced form)
    print(bin_to_h_unreduced("0000000010001110"))
    print(h_to_bin_unreduced("AAABBBA"))
    print()

    # Conversion between binary strings and reduced h
    print(bin_to_h_reduced("0000000010001110"))  # This produces something completely different from bin_to_h_unreduced
    print(h_to_bin_reduced("ABAABB"))  # This is not implemented yet, but from the enumeration below we know it should give 41

    # Conversion between h and matrix
    print(h_to_mat("ABAABB"))

    # Enumeration of reduced h
    for i in range(2, 100):
        s = bin_to_h_reduced(int_to_bin(i))
        print(str(i).rjust(2)+": "+s)
    '''

def h_to_mat(s, mod=0):
    mA = A
    mB = B
    if mod > 1:
        mA %= mod
        mB %= mod
    result = I3
    for i in range(len(s)):
        m = mA if s[i] == "A" else mB
        result = result*m
        if mod > 1: result %= mod
    return result

def int_to_bin(n):
    return np.binary_repr(n)

def bin_to_int(b):
    a = np.array([int(d) for d in list(b)])
    return a.dot(2**np.arange(a.size)[::-1])

def bin_to_h_unreduced(b, alphabet=True):
    b = b.lstrip("0")  # Remove any leading zeroes
    if len(b) == 0: return None
    b = b[1:]  # Remove the leading 1
    if alphabet: b = b.replace("0", "A").replace("1", "B")
    return b

def h_to_bin_unreduced(s, alphabet=True):
    if alphabet: s = s.replace("A", "0").replace("B", "1")
    return "1"+s

def int_to_h(n, alphabet=True):
    return bin_to_h_unreduced(int_to_bin(n), alphabet)

def h_to_int(s, alphabet=True):
    return bin_to_int(h_to_bin_unreduced(s, alphabet))

def bin_to_h_reduced(b):  # Turns out this doesn't work.
    b = b.lstrip("0")  # Remove any leading zeroes
    b = b[1:]  # Remove the leading 1
    result = ""
    i = 0
    while i < len(b):
        previous = ("xxxxxxxxxxxxxxx"+result)[-15:]  # Get the 15 previous factors, substituting x for any empty spots
        next_factor = None
        if previous[-2:] == "AA": next_factor = "B"
        elif previous[-2:] == "BB": next_factor = "A"
        elif previous[-7:] == "ABABABA": next_factor = "A"
        elif previous[-7:] == "BABABAB": next_factor = "B"
        elif previous == "AABBAABBAABBAAB": next_factor = "A"
        elif previous == "BBAABBAABBAABBA": next_factor = "B"

        if next_factor is None:
            result += ("A" if b[i] == "0" else "B")
            i += 1
        else:
            result += next_factor
    return result

def h_to_bin_reduced(s):
    pass  # I'll leave this to Michaela (or future Gabriel) to implement

if __name__ == "__main__":
    main()