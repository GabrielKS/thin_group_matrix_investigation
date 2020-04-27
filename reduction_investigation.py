from common import *
import conversion
import time
import copy
import math
import atexit
import os
import pickle

# OUTPUT SETTINGS
max_length = 30  # Maximum word length to examine
print_times_length = True  # Whether or not to print how much time the steps take
min_length_to_save = 15  # Minimum length to save the output files every iteration
output_dir = "output"  # Output folder name

# INTERMEDIATE SETTINGS
cache_length = math.ceil(max_length/2)

# INTERMEDIATE DATA STRUCTURES
words_cache = [None]  # Element 0 of words is None. Element i of words for i>0 is a dictionary containing information about the word corresponding to that index
timers = {"population_time": 0, "analysis_time": 0, "summarization_time": 0}
log_strings = {"population_time": "", "analysis_time": "", "summarization_time": "", "results_summary": ""}
states = {"length_calculated": -1}
unique_matrices_hashed = np.empty((0))  # A list of the hashes of all the matrices in unique_results_all, for faster computation

# OUTPUT DATA STRUCTURES
unique_results_all = []  # Array of dictionaries per unique matrix {"mat": NumPy_matrix_representing_this_h, "refs": all_integer_representations_of_this_h}
unique_results_length = []  # Similar to unique_results_all but one for each length, where the entries for each length do not contain any references to words of other lengths

def main():
    atexit.register(save_output)  # Save output whenever the program exits
    for length in range(cache_length+1):
        populate(length)

    for length in range(max_length+1):
        analyze(length)
        summarize(length)
        if length >= min_length_to_save: save_output()

def populate(length):  # Populate words with all the words of length length, assuming populate(length-1) has already been called for all shorter length
    t1 = time.perf_counter()

    assert len(words_cache) == length_to_ref(length)
    these_words = [None for i in range(length_to_ref(length))]
    for i in range(0, length_to_ref(length)):
        these_words[i] = conversion.h_to_mat(conversion.int_to_h(length_to_ref(length)+i))
    words_cache.extend(these_words)
    
    t2 = time.perf_counter()

def analyze(length):  # Analyze words of length length, assuming analyze(length-1) has already been called for all shorter lengths but not this one
    t1 = time.perf_counter()
    start_ref = length_to_ref(length)
    end_ref = length_to_ref(length+1)
    these_unique_results = []
    these_unique_matrices_hashed = np.empty((0))
    t_pop = 0

    for ref in range(start_ref, end_ref):  # Populate these_unique_results and these_unique_matrices_hashed
        t_pop_1 = time.perf_counter()
        mat = int_to_mat_cached(ref)
        t_pop += time.perf_counter()-t_pop_1

        this_hash = np.array([hash(np.asarray(mat).tobytes())])  # Get a hash for streamlined comparison
        found = False  # We now compare to what we have in these_unique_results so far
        same_length_comparison = np.equal(this_hash, these_unique_matrices_hashed)
        for i, v in enumerate(same_length_comparison):
            if v:  # If we found a match, we add a reference to this one and move on
                assert np.equal(mat, these_unique_results[i]["mat"]).all(), "HASH COLLISION"  # Since we're dealing with such enormous amounts of data, a hash collision is possible. If this ends up being a problem, it won't be hard to get around (we'll just do this check every time we find a probable match), but I'd like to know if it happens first.
                found = True
                these_unique_results[i]["refs"].append(ref)
                break  # There is only one match
        if not found:  # If we didn't find it, we create a new entry in our results and also add it to the streamlined data structure
            these_unique_results.append({"mat": mat, "refs": [ref]})
            these_unique_matrices_hashed = np.concatenate((these_unique_matrices_hashed, this_hash))  # I think this is inefficient, but I don't see a good alternative (and it's not done with *that* much data)
    unique_results_length.append(these_unique_results)  # Save the per-length results globally (no need to keep the streamlined matrices per length)

    for this_result in these_unique_results:  # Now we search the results for shorter lengths and look for matches there
        this_hash = np.array([hash(np.asarray(this_result["mat"]).tobytes())])
        found = False
        existing_comparison = np.equal(this_hash, unique_matrices_hashed)  # We require that unique_matrices_all contain all unique matrices with reduced forms of shorter length
        for i, v in enumerate(existing_comparison):
            if v:
                assert np.equal(this_result["mat"], unique_results_all[i]["mat"]).all(), "HASH COLLISION 2"  # See above
                found = True
                unique_results_all[i]["refs"].extend(this_result["refs"])  # We add the refs of the current length to the list of existing refs (OK that we don't keep this separated by length because it's easy to test a ref for length)
                break
        if not found:
            unique_results_all.append({"mat": this_result["mat"], "refs": this_result["refs"]})
            unique_matrices_hashed.resize((unique_matrices_hashed.shape[0]+1))  # Expand unique_matrices_all by the amount we need to append to it (trying to avoid using the global keyword)
            unique_matrices_hashed[unique_matrices_hashed.shape[0]-1] = this_hash  # Same efficiency concerns as above; we're basically appending in-place

    timers["population_time"] += t_pop
    log(format_ratio("Population time (length="+str(length)+")", t_pop, timers["population_time"]), "population_time", print_times_length)

    t2 = time.perf_counter()
    timers["analysis_time"] += t2-t1-t_pop
    log(format_ratio("Analysis time (length="+str(length)+")", t2-t1-t_pop, timers["analysis_time"]), "analysis_time", print_times_length)

def summarize(length):  # Computes summary statistics for words of length length and all words up to length length. Prepare for lots of dense one-liners.
    t1 = time.perf_counter()

    these_words = words_cache[length_to_ref(length):length_to_ref(length+1)]
    these_unique_results = unique_results_length[length]

    total_words_all = length_to_ref(length+1)-1
    total_words_length = length_to_ref(length)

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

    states["length_calculated"] = length

    t2 = time.perf_counter()
    timers["summarization_time"] += t2-t1
    log(format_ratio("Summarization time (length="+str(length)+")", t2-t1, timers["summarization_time"]), "summarization_time", print_times_length)
    if print_times_length: print()
    
    log("Length "+str(length)+":", "results_summary")
    log(format_ratio("Total", total_words_length, total_words_all), "results_summary")
    log(format_ratio("Reduced", reduced_words_length, reduced_words_all), "results_summary")
    log(format_ratio("Same-length unique", same_length_unique_length, same_length_unique_all), "results_summary")
    log(format_ratio("Same-and-shorter unique", same_and_shorter_unique_length, same_and_shorter_unique_all), "results_summary")
    log(format_ratio("Reduced nonunique", reduced_nonunique_length, reduced_nonunique_all), "results_summary")
    log("", "results_summary")

def save_output():
    # Put together the text output
    output_log = "length_calculated="+str(states["length_calculated"])+"\n\n"+"\n\n".join(["LOG "+log_string_key+":\n"+log_strings[log_string_key] for log_string_key in log_strings])
    output_results_all = results_to_string(unique_results_all, "all")
    output_results_length = "\n\n".join([results_to_string(unique_result_length, "length="+str(i)) for i, unique_result_length in enumerate(unique_results_length)])

    # Write the text output
    write_output(output_log, "output_log.txt")
    write_output(output_results_all, "output_results_all.txt")
    write_output(output_results_length, "output_results_length.txt")

    # Pickle the important variables so they can be further analyzed programmatically
    pickle_data(unique_results_all, "results_all.pickle")
    pickle_data(unique_results_length, "results_length.pickle")

def write_output(output, filename):
    with open(os.path.join(output_dir, filename), "w") as output_file:
        output_file.write(output)

def pickle_data(data, filename):
    with open(os.path.join(output_dir, filename), "wb") as data_file:
        pickle.dump(data, data_file)

def results_to_string(results, label):
    matrices = "MATRICES ("+label+"):\n"
    refs = "REFS ("+label+"):\n"
    for i, result in enumerate(results):
        matrices += str(i+1)+" (CRR="+str(result["refs"][0])+"):\n"+str(result["mat"])+"\n"  # CRR: canonical reduced ref(erence)
        refs += str(i+1)+": "+str(result["refs"])+"\n"
    return matrices+"\n"+refs

def count_true(arr):  # Count the number of True values in a boolean array
    return np.count_nonzero(arr)

def length_to_ref(length):  # The first reference to a word of length length will be at length_to_ref(length). Also, the number of words of length length will be length_to_ref(length).
    return 2**length

def ref_to_length(ref):  # The length of the word represented by ref
    return int(math.log2(ref))  # Clearer than messing around with bits

def format_ratio(label, num, denom):  # Nicely formats a label and a ratio
    return label+": "+str(num)+"/"+str(denom)

def log(output, log_string, print_output=True):
    if print_output: print(output)
    log_strings[log_string] += output+"\n"

def int_to_mat_cached(ref):
    h = conversion.int_to_h(ref, alphabet=False)  # TODO do this more mathematically if it's a major source of slowdown
    ref1 = conversion.h_to_int(h[:cache_length], alphabet=False)
    ref2 = conversion.h_to_int(h[cache_length:], alphabet=False)
    return words_cache[ref1]*words_cache[ref2]

if __name__ == "__main__":
    main()