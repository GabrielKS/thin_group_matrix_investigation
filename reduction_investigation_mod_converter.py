import reduction_investigation as investigator
import reduction_investigation_viewer as viewer
import numpy as np
from common import *

modulo = 7
write_files = True
update_frequency = 100

def main():
    print_program_start()
    print("LOADING")
    viewer.load("output 33")
    print(viewer.max_length)
    viewer.unique_results_length = [None]*(viewer.max_length+1)  # In case we're not loading these
    print("\nANALYZING")
    analyze_mods(modulo)
    print("\nSUMMARIZING\n")
    for length in range(viewer.max_length+1):
        investigator.summarize(length)
    print("\nSAVING")
    if write_files: investigator.save_output(force=True)
    print("\nDONE")

def analyze_mods(modulo):
    for i in range(len(viewer.unique_results_all)):
        viewer.unique_results_all[i]["mat"] %= modulo  # Modulo everything
    all_matrices = np.array([result["mat"] for result in viewer.unique_results_all])
    all_matrices_hashes = np.array([hash_3x3(result["mat"]) for result in viewer.unique_results_all])
    for i in range(len(viewer.unique_results_all)):
        if i % (len(viewer.unique_results_all)//update_frequency) == 0: print("{:.2%}".format(i/len(viewer.unique_results_all)))  # Print a status update every update_frequency-th of the way
        j = investigator.find_first_equal_hash(viewer.unique_results_all[i]["mat"], all_matrices, hash_3x3(viewer.unique_results_all[i]["mat"]), all_matrices_hashes)
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