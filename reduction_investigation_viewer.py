from common import *
import os
import pickle

max_length = None
unique_results_all = []
unique_results_length = []

def main():
    print_program_start()
    load("output 33")
    # print(max_length)
    # print(unique_results_all)
    # print(unique_results_length)

    # for i, result in enumerate(unique_results_all):
    #     if len(result["refs"]) > 300: print(str(i)+": "+str(len(result["refs"])))

    # print(length_to_ref(max_length+1)-1)
    # print(sum([len(unique_results_all[i]["refs"]) for i in range(len(unique_results_all))]))

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
    for i in range (len(unique_results_all)):
        crr = unique_results_all[i]["refs"][0]
        matrix = unique_results_all[i]["mat"]
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
    global max_length, unique_results_all, unique_results_length
    with open(os.path.join(output_dir, "output_log.txt")) as output_log:
        max_length = int(output_log.readline().split("=")[1])
    unique_results_all = unpickle_data(os.path.join(output_dir, "results_all.pickle"))
    # unique_results_length = unpickle_data(os.path.join(output_dir, "results_length.pickle"))

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
        for entry in unique_results_all:
            if ref_to_length(entry["refs"][0]) == i:
                these_entries = entry["mat"].reshape(1, 9).tolist()[0]
                for j in range(9):
                    if extreme_entries[j] is None or comparator(abs(these_entries[j]), abs(extreme_entries[j])):
                        extreme_entries[j] = abs(these_entries[j])
        results.append(np.matrix(extreme_entries).reshape(3,3))
    return results

if __name__ == "__main__":
    main()