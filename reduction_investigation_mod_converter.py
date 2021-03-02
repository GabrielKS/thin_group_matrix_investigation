import reduction_investigation as investigator
import reduction_investigation_viewer as viewer
import numpy as np
from common import *

modulo = 5
write_files = True
update_frequency = 100

def hash_3x3(mat):  # A substitute for hash(mat.tobytes())
    return hash((mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2]))

#@numba.jit(numba_dtype(numba.types.Array(numba_dtype, 2, "A", readonly=True), numba.types.Array(numba_dtype, 3, "A", readonly=True), numba_dtype, numba.types.Array(numba_dtype, 1, "A", readonly=True)), nopython=True)  # In this case, if we don't manually specify type, Numba actually slows things down.
def find_first_equal_hash(elem, elem_list, elem_hash, elem_hash_list):  # Returns the index of the first value of elem_list equal to elem, or -1 if there is no match
    for i in range((len(elem_list))):
        if elem_hash == elem_hash_list[i] and equals_lazy_3x3(elem, elem_list[i,:,:]): return i
    return -1

#@numba.jit(nopython=True)
def equals_lazy_3x3(a, b):  # Equivalent to np.equal(a, b).all()
    return a[0,0]==b[0,0] and a[0,1]==b[0,1] and a[0,2]==b[0,2] and a[1,0]==b[1,0] and a[1,1]==b[1,1] and a[1,2]==b[1,2] and a[2,0]==b[2,0] and a[2,1]==b[2,1] and a[2,2]==b[2,2]

def main():
    print("LOADING")
    viewer.load("output")
    print(viewer.max_length)
    viewer.unique_results_length = [None]*(viewer.max_length+1)  # In case we're not loading these
    print("\nANALYZING")
    analyze_mods(modulo)
    print("\nSUMMARIZING\n")
    for length in range(viewer.max_length+1):
        investigator.summarize(length)
    print("\nSAVING")
    if write_files: investigator.save_output(label="_"+str(modulo))
    print("\nDONE")

def analyze_mods(modulo):
    for i in range(len(viewer.unique_results_all)):
        viewer.unique_results_all[i]["mat"] %= modulo  # Modulo everything
    all_matrices = np.array([result["mat"] for result in viewer.unique_results_all])
    all_matrices_hashes = np.array([hash_3x3(result["mat"]) for result in viewer.unique_results_all])
    for i in range(len(viewer.unique_results_all)):
        if i % (len(viewer.unique_results_all)//update_frequency) == 0: print("{:.2%}".format(i/len(viewer.unique_results_all)))  # Print a status update every update_frequency-th of the way
        j = find_first_equal_hash(viewer.unique_results_all[i]["mat"], all_matrices, hash_3x3(viewer.unique_results_all[i]["mat"]), all_matrices_hashes)
        if j < i:  # If we find an earlier match...
            if ref_to_length(viewer.unique_results_all[j]["refs"][0]) == ref_to_length(viewer.unique_results_all[i]["refs"][0]): viewer.unique_results_all[j]["refs"].extend(viewer.unique_results_all[i]["refs"])  # ...put the refs there if it's the same length...
            viewer.unique_results_all[i]["delete"] = True  # ...mark this entry for deletion
    viewer.unique_results_all = [result for result in viewer.unique_results_all if "delete" not in result]

    previous_end = 0  # Partition unique_results_all into unique_results_length
    for length in range(viewer.max_length+1):
        this_end = len(viewer.unique_results_all)
        for i in range(previous_end, len(viewer.unique_results_all)):
            if ref_to_length(viewer.unique_results_all[i]["refs"][0]) > length:
                this_end = i
                break
        viewer.unique_results_length[length] = viewer.unique_results_all[previous_end:this_end]
        previous_end = this_end
    
    investigator.unique_results_length = viewer.unique_results_length  # Copy the results over to investigator so summarize() can see them
    investigator.unique_results_all = viewer.unique_results_all
    investigator.modulo = modulo

if __name__ == "__main__":
    main()


