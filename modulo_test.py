from common import *
import conversion

def main():
    # for m in gen_mod_mat(3): print(m)
    results = universal_modulo_test()
    print("We have results!")
    # print(results)
    universal_modulo_output(results, print_matrices=True)


def universal_modulo_test(skip_bottom=False):  # Highly inefficient. For one thing, the loops could be cleaned up and some advantage could probably be taken of NumPy's lower-level loops; for another, this workflow requires storing everything in memory between when it is calculated and when it is interpreted.
    times = 1000  # How many matrices to test (increasing this will make things slower linearly)
    h_length = 10  # How long to make the word (increasing this will make things slower roughly linearly)
    max_mod = 2  # Maximum modulo to test (increasing this will make things slower *very very exponentially*)
    results = [{}]*(max_mod-1)  # Results is an array of dictionaries. The array index corresponds to 2 minus the mod under consideration (0 means mod 2, 1 means mod 3, etc.). The dictionary key is a matrix of ints >= 0 and < the mod under consideration. The dictionary value is a word that generates the key when turned into a matrix and taken mod the mod under consideration.
    for i in range(times):
        if (i % (times//100) == 0): print(i)
        # print(i)
        h_str = random_H_str(h_length)
        h_mat = conversion.h_to_mat(h_str)
        for mod in range(2, max_mod+1):
            for mod_mat in gen_mod_mat(mod, skip_bottom=skip_bottom):
                remainder = np.mod(h_mat, mod)
                match = np.equal(remainder, mod_mat)[0:2, :].all() if skip_bottom else np.equal(remainder, mod_mat).all()  # We might only care about the first two rows
                if match:
                    results[mod-2][mat_to_key(mod_mat)] = h_str
    return results

def universal_modulo_output(results, skip_bottom=False, print_matrices=False):
    for i,result in enumerate(results):
        mod = i+2
        found_count = 0
        not_found_count = 0
        for mod_mat in gen_mod_mat(mod):
            found = mat_to_key(mod_mat) in result
            # # Things we already know:
            # if mod % 3 == 0 and not(mod_mat[2,0]%3 == 0 and mod_mat[2,1]%3 == 0 and mod_mat[2,2]%3 == 1):  # Bottom row mod 3 is 0,0,1
            #     assert not found, "VIOLATES BOTTOM ROW 001:\n"+str(mod_mat)
            #     continue  # No need to say we didn't find it
            # I'm not even processing the bottom row anymore. But this is the spot to add any subtler rules we find so we can filter the output down to things we care about.

            if found:
                if print_matrices: print("found mod "+str(mod)+":\n"+str(mod_mat[0:2,:] if skip_bottom else mod_mat)+"\n")
                found_count += 1
            else:
                if print_matrices: print("NOT FOUND mod "+str(mod)+":\n"+str(mod_mat[0:2,:] if skip_bottom else mod_mat)+"\n")
                not_found_count += 1
        print("total found in mod "+str(mod)+": "+str(found_count))
        print("total NOT FOUND in mod "+str(mod)+": "+str(not_found_count))

def mat_to_key(mat):
    return tuple([tuple(row) for row in mat])


def gen_mod_mat(mod, skip_bottom=False):  # Generates all the possibilities for a 3x3 matrix mod <mod> where the bottom row is always [[0,0,0]]. Use like: for m in gen_mod_mat(3): print(m). Heavily hardcoded and inelegant.
    for i in range(mod**(6 if skip_bottom else 9)):
        n = i
        entries = []
        for j in range(6 if skip_bottom else 9):
            entries.append(n % mod)
            n //= mod
        yield np.array(entries+[0,0,0] if skip_bottom else entries).reshape(3, 3)


if __name__ == "__main__":
    main()