from common import *
import conversion
import time
import copy
import math

# SETTINGS
max_length = 15
print_times_length = False

# INTERMEDIATE DATA STRUCTURES
words = [None]  # Element 0 of words is None. Element i of words for i>0 is a dictionary containing information about the word corresponding to that index
timers = {"population_time": 0, "analysis_time": 0, "summarization_time": 0}
unique_matrices_all = np.empty((0,3,3))  # All the matrices in unique_results_all but in 3D form, for faster computation

# OUTPUT DATA STRUCTURES
unique_results_all = []  # Array of dictionaries per unique matrix {"mat": NumPy_matrix_representing_this_h, "refs": all_integer_representations_of_this_h}
unique_results_length = []  # Similar to unique_results_all but one for each length, where the entries for each length do not contain any references to words of other lengths

def main():
    for length in range(max_length+1):
        populate(length)
        analyze(length)
        summarize(length)
        # time.sleep(1.5)
        print()
    print()
    print("Total population time: "+str(timers["population_time"]))
    print("Total analysis time: "+str(timers["analysis_time"]))
    print("Total summarization time: "+str(timers["summarization_time"]))
    print()
    with open("output.txt", "w") as output_file:
        output_file.write("\n".join([str(r["refs"]) for r in unique_results_all]))


def populate(length):  # Populate words with all the words of length length, assuming populate(length-1) has already been called for all shorter length
    t1 = time.perf_counter()

    assert len(words) == length_to_ref(length)
    these_words = [{} for i in range(length_to_ref(length))]
    for i in range(0, length_to_ref(length)):
        these_words[i]["ref"] = length_to_ref(length)+i
        these_words[i]["mat"] = conversion.h_to_mat(conversion.int_to_h(these_words[i]["ref"]))
    words.extend(these_words)
    
    t2 = time.perf_counter()
    timers["population_time"] += t2-t1
    if print_times_length: print("Population time (length="+str(length)+"): "+str(t2-t1))

def analyze(length):  # Analyze words of length length, assuming analyze(length-1) has already been called for all shorter lengths but not this one
    t1 = time.perf_counter()
    
    these_words = words[length_to_ref(length):length_to_ref(length+1)]  # These are all the words of length length

    these_unique_results = []
    these_unique_matrices = np.empty((0,3,3))
    
    for this_word in these_words:  # Populate these_unique_results and these_unique_matrices
        this_nd = np.asarray(this_word["mat"])  # Convert from NumPy matrix to NumPy ndarray for the streamlined comparison
        found = False  # We now compare to what we have in these_unique_results so far
        same_length_comparison = np.equal(this_nd, these_unique_matrices).all(axis=1).all(axis=1)
        for i, v in enumerate(same_length_comparison):
            if v:  # If we found a match, we add a reference to this one and move on
                found = True
                these_unique_results[i]["refs"].append(this_word["ref"])
                break  # There is only one match
        if not found:  # If we didn't find it, we create a new entry in our results and also add it to the streamlined data structure
            these_unique_results.append({"mat": this_word["mat"], "refs": [this_word["ref"]]})
            these_unique_matrices = np.concatenate((these_unique_matrices, np.reshape(this_nd, (1, *this_nd.shape))))  # I think this is inefficient, but I don't see a good alternative (and it's not done with *that* much data)
    unique_results_length.append(these_unique_results)  # Save the per-length results globally (no need to keep the streamlined matrices per length)

    for this_result in these_unique_results:  # Now we search the results for shorter lengths and look for matches there
        this_nd = np.asarray(this_result["mat"])
        found = False
        existing_comparison = np.equal(this_nd, unique_matrices_all).all(axis=1).all(axis=1)  # We require that unique_matrices_all contain all unique matrices with reduced forms of shorter length
        for i, v in enumerate(existing_comparison):
            if v:
                found = True
                unique_results_all[i]["refs"].extend(this_result["refs"])  # We add the refs of the current length to the list of existing refs (OK that we don't keep this separated by length because it's easy to test a ref for length)
                break
        if not found:
            unique_results_all.append({"mat": this_result["mat"], "refs": this_result["refs"]})
            unique_matrices_all.resize((lambda a,b,c: (a+1,b,c))(*unique_matrices_all.shape))  # Expand unique_matrices_all by the amount we need to append to it (trying to avoid using the global keyword)
            unique_matrices_all[unique_matrices_all.shape[0]-1, :, :] = this_nd  # Same efficiency concerns as above; we're basically appending in-place

    t2 = time.perf_counter()
    timers["analysis_time"] += t2-t1
    if print_times_length: print("Analysis time (length="+str(length)+"): "+str(t2-t1))

def summarize(length):  # Computes summary statistics for words of length length and all words up to length length. Prepare for lots of dense one-liners.
    t1 = time.perf_counter()

    these_words = words[length_to_ref(length):length_to_ref(length+1)]
    these_unique_results = unique_results_length[length]

    total_words_all = len(words)-1
    total_words_length = len(these_words)

    canonically_reduced_words_all = len(unique_results_all)  # Same number of canonically reduced words as unique results
    canonically_reduced_words_length = count_true([ref_to_length(this_result["refs"][0]) == length for this_result in unique_results_all])  # A word of length length is canonically reduced if it is the first ref for its result

    reduced_words_all = sum([
        count_true([
            ref_to_length(ref) == ref_to_length(this_result["refs"][0])
            for ref in this_result["refs"]])
        for this_result in unique_results_all])  # A word is reduced if it has the same length as the canonically reduced form of its matrix
    reduced_words_length = sum([
        0 if ref_to_length(this_result["refs"][0]) != length else
        count_true([
            ref_to_length(ref) == length
            for ref in this_result["refs"]])
        for this_result in unique_results_all])  # A word of length length is reduced if the canonically reduced form of its matrix is of length length

    def same_length_unique(l):
        return count_true([len(this_result["refs"]) == 1 for this_result in unique_results_length[l]])  # A word of length length is same-length unique if it is the only ref for its per-length result
    same_length_unique_all = sum([same_length_unique(l) for l in range(length+1)])  # Clearly inefficient; optimize by storing previous results if necessary
    same_length_unique_length = same_length_unique(length)

    same_and_shorter_unique_all = count_true([
        count_true([
            ref_to_length(ref) == ref_to_length(this_result["refs"][0])
            for ref in this_result["refs"]]) == 1
        for this_result in unique_results_all])  # A word is same-and-shorter unique if it is canonically reduced and there are no other reduced forms of its matrix
    same_and_shorter_unique_length = count_true([
        False if ref_to_length(this_result["refs"][0]) != length else
        count_true([
            ref_to_length(ref) == length
            for ref in this_result["refs"]]) == 1
        for this_result in unique_results_all])  # A word of length length is same-and-shorter unique if it is of length length and is canonically reduced and there are no other reduced forms of its matrix
    
    reduced_nonunique_all = sum([
        (lambda n: 0 if n == 1 else n)
            (count_true([
                ref_to_length(ref) == ref_to_length(this_result["refs"][0])
                for ref in this_result["refs"]]))
        for this_result in unique_results_all])  # A word is reduced nonunique if it has the same length as the canonically reduced form of its matrix and is not the only one to satisfy this property

    reduced_nonunique_length = sum([
        (lambda n: 0 if n == 1 else n)
            (0 if ref_to_length(this_result["refs"][0]) != length else
            count_true([
                ref_to_length(ref) == length
                for ref in this_result["refs"]]))
        for this_result in unique_results_all])  # A word is reduced nonunique if it has the same length as the canonically reduced form of its matrix, if that is length length, and it is not the only one to satisfy this property

    t2 = time.perf_counter()
    timers["summarization_time"] += t2-t1
    if print_times_length:
        print("Summarization time (length="+str(length)+"): "+str(t2-t1))
        print()
    
    print("Length "+str(length)+":")
    print_labeled_ratio("Total", total_words_length, total_words_all)
    print_labeled_ratio("Reduced", reduced_words_length, reduced_words_all)
    print_labeled_ratio("Same-length unique", same_length_unique_length, same_length_unique_all)
    print_labeled_ratio("Same-and-shorter unique", same_and_shorter_unique_length, same_and_shorter_unique_all)
    print_labeled_ratio("Reduced nonunique", reduced_nonunique_length, reduced_nonunique_all)

def count_true(arr):  # Count the number of True values in a boolean array
    return np.count_nonzero(arr)

def length_to_ref(length):  # The first reference to a word of length length will be at length_to_ref(length). Also, the number of words of length length will be length_to_ref(length).
    return 2**length

def ref_to_length(ref):  # The length of the word represented by ref
    return int(math.log2(ref))  # Clearer than messing around with bits

def print_labeled_ratio(label, num, denom):  # Nicely prints a label and a ratio
    print(label+": "+str(num)+"/"+str(denom))

if __name__ == "__main__":
    main()