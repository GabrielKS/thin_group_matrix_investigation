from common import *
import conversion

length = 8

def main():
    found = False
    for i in range (2**length, 2**(length+1)):
        matrix = conversion.h_to_mat(conversion.ref_to_h(i))
        if found:
            break
        for j in range (0,3):
            if found:
                break
            for k in range (0,3):
                if matrix[j,k] == 0:
                    print(i)
                    print(matrix)
                    found = True
                    break



if __name__ == "__main__":
    main()