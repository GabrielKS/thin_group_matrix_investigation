from common import *
import os
import pickle

max_length = None
unique_matrices = np.zeros((0,3,3), dtype=dtype)
unique_refs = []

def main():
    print_program_start()
    load("output")
    # print(max_length)
    # print(unique_matrices)
    # print(unique_refs)

    # for i, result in enumerate(unique_refs):
    #     if len(result) > 300: print(str(i)+": "+str(len(result)))

    # print(length_to_ref(max_length+1)-1)
    # print(sum([len(unique_refs[i]) for i in range(len(unique_refs))]))

    # findEntry(15,1)

    # for i, m in enumerate(zip(matrices_of_extreme_entries("min"), matrices_of_extreme_entries("max"))):  # Cool but not actually the ideal format...
    #     print(str(i)+"\n"+str(m[0])+"\n"+str(m[1])+"\n")

    print("Matrices of least entries")
    for i, m in enumerate(matrices_of_extreme_entries("min")):
        print(str(i)+"\n"+str(m)+"\n")

    print("Matrices of greatest entries")
    for i, m in enumerate(matrices_of_extreme_entries("max")):
        print(str(i)+"\n"+str(m)+"\n")

def findEntry(length, entry):
    found = False
    for i in range (len(unique_refs)):
        crr = unique_refs[i][0]
        matrix = unique_matrices[i]
        if crr >= 2**length and crr < 2**(length+1):
            if found:
                break
            for j in range (0,3):
                if found:
                    break
                for k in range (0,3):
                    if matrix[j,k] == entry:
                        print(crr)
                        print(matrix)
                        found = True
                        break
    for m in matrices_of_extreme_entries("min"):
        print(str(m)+"\n")

def load(output_dir):
    global max_length, unique_matrices, unique_refs
    with open(os.path.join(output_dir, "output_log.txt")) as output_log:
        max_length = int(output_log.readline().split("=")[1])
    unique_matrices = np.load(os.path.join(output_dir, "unique_matrices.npy"), allow_pickle=False)
    refs_lengths = np.load(os.path.join(output_dir, "refs_lengths.npy"), allow_pickle=False)
    refs_stacked = np.load(os.path.join(output_dir, "refs_stacked.npy"), allow_pickle=False)
    unique_refs = np.split(refs_stacked, refs_lengths)  # This approach inspired by https://tonysyu.github.io/ragged-arrays.html

def unpickle_data(filename):
    data = None
    with open(filename, "rb") as data_file:
        data = pickle.load(data_file)
    return data

def matrices_of_extreme_entries(criterion):    # Returns one matrix per length where each entry is the absolute maximum/minimum of the entries in that spot in all the matrices of that length
    results = []
    comparator = (lambda x, y: x < y) if criterion == "min" else (lambda x, y: x > y) if criterion == "max" else None
    for i in range(max_length+1):
        extreme_entries = [None]*9
        for entry in zip(unique_matrices, unique_refs):
            if ref_to_length(entry[1][0]) == i:
                these_entries = entry[0].reshape(1, 9).tolist()[0]
                for j in range(9):
                    if extreme_entries[j] is None or comparator(abs(these_entries[j]), abs(extreme_entries[j])):
                        extreme_entries[j] = abs(these_entries[j])
        results.append(np.matrix(extreme_entries).reshape(3,3))
    return results

if __name__ == "__main__":
    main()