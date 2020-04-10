from common import *

def main():
    print(h_str_to_mat("ABAABB"))

def h_str_to_mat(s):
    result = I3
    for i in range(len(s)):
        m = A if s[i] == "A" else B
        result = result*m
    return result

if __name__ == "__main__":
    main()