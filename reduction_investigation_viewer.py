from common import *
import os
import pickle

max_length = None
unique_results_all = None
unique_results_length = None

def main():
    load("output")
    # print(max_length)
    # print(unique_results_all)
    # print(unique_results_length)

    # for m in matrices_of_least_entries():
    #     print(str(m)+"\n")

    # for i, result in enumerate(unique_results_all):
    #     if len(result["refs"]) > 300: print(str(i)+": "+str(len(result["refs"])))

    print(length_to_ref(max_length+1)-1)
    print(sum([len(unique_results_all[i]["refs"]) for i in range(len(unique_results_all))]))

    # findEntry(15,1)

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
    for m in matrices_of_least_entries():
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

def matrices_of_least_entries():    # Returns one matrix per length where each entry is the absolute minimum of the entries in that spot in all the matrices of that length
    results = []
    for i in range(max_length):
        least_entries = [None]*9
        for entry in unique_results_all:
            if ref_to_length(entry["refs"][0]) == i:
                these_entries = entry["mat"].reshape(1, 9).tolist()[0]
                for j in range(9):
                    if least_entries[j] is None or abs(these_entries[j]) < abs(least_entries[j]):
                        least_entries[j] = abs(these_entries[j])
        results.append(np.matrix(least_entries).reshape(3,3))
    return results

if __name__ == "__main__":
    main()