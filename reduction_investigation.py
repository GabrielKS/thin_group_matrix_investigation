from common import *
import conversion
import time
import copy
import math
import atexit
import os
import pickle
import numba

# OUTPUT SETTINGS
max_length = 28  # Maximum word length to examine
modulo = 0  # 0 for non-modular arithmetic, >1 to multiply and compare matrices using modular arithmetic
print_times_length = True  # Whether or not to print how much time the steps take
min_length_to_save = 26  # Minimum length to save the output files every iteration
output_prefix = "output"  # First part of folder name (second part is for the mod, if any)
write_files = False  # Whether or not to save output to files
max_refs_per_length = -1  # Maximum number of refs per length to store (all the others are discarded); -1 for keep them all
only_reduced = True

# INTERMEDIATE SETTINGS
cache_length = math.ceil(max_length/2)  # Maximum word length to cache matrices for (must be at least half of max_length)
max_warnings = 10  # Maximum number of warnings per word length about matrix entries being very large
chunk_size = 2**20  # Number of previous refs to process in each chunk


# INTERMEDIATE DATA STRUCTURES
words_cache = np.zeros((length_to_ref(cache_length+1), 3, 3), dtype=dtype)  # Element 0 of words is a 3x3 array of zeroes, a stand-in for None. Element i of words for i>0 is the matrix corresponding to ref i
timers = {"analysis_time": 0, "summarization_time": 0}
log_strings = {"analysis_time": "", "summarization_time": "", "results_summary": ""}
states = {"length_calculated": -1, "length_saved": -1, "warnings": 0}
totals = {"total_words": 0, "reduced_words": 0, "canonically_reduced_words": 0, "same_length_unique": 0, "same_and_shorter_unique": 0, "reduced_nonunique": 0}
unique_matrices = np.array([O3])  # A list of all the matrices in unique_results_all, for faster computation. We initialize with an entry because Numba doesn't like empty arrays.
unique_matrices_hashed = np.array([0], dtype=hashtype)  # A hashed version of unique_matrices, for even faster computation
unique_matrices_length = numba.typed.List()
unique_matrices_length_hashed = numba.typed.List()
unique_refs_length = numba.typed.List()

# OUTPUT DATA STRUCTURES
unique_results_all = []  # Array of dictionaries per unique matrix {"mat": NumPy_matrix_representing_this_h, "refs": all_integer_representations_of_this_h}
unique_results_length = []  # Similar to unique_results_all but one for each length, where the entries for each length do not contain any references to words of other lengths

def main():
    print_program_start()
    atexit.register(save_output)  # Save output whenever the program exits
    for length in range(cache_length+1):
        populate(length)

    for length in range(max_length+1):
        t1 = time.perf_counter()
        analyze_new(length)
        t2 = time.perf_counter()
        timers["analysis_time"] += t2-t1
        log(format_ratio("Analysis time (length="+str(length)+")", t2-t1, timers["analysis_time"]), "analysis_time", print_times_length)
        
        summarize(length)
        
        if length >= min_length_to_save: save_output()

# @numba.jit(nopython=True)  # TODO: Numba-ify this -- involves Numba-ifying (or most likely working around) h_to_mat -- low priority as it is not a bottleneck
def populate(length):  # Populate words with all the words of length length, assuming populate(length-1) has already been called for all shorter length
    for i in range(length_to_ref(length), length_to_ref(length+1)):
        words_cache[i,:,:] = conversion.h_to_mat(conversion.ref_to_h(i), mod=modulo)

def analyze_new(length):
    assert only_reduced  # The new algorithm only finds reduced words
    if length == 0:  # Simplest to just hard-code the base case
        current_matrices = np.array([I3])
        current_matrices_hashed = np.array([hash_3x3(current_matrices[0])])
        current_refs = np.array([1])
    else: 
        previous_matrices = unique_matrices_length[length-1]
        previous_matrices_hashed = unique_matrices_length_hashed[length-1]
        previous_refs = unique_refs_length[length-1]
        t1 = time.perf_counter()
        current_matrices, current_matrices_hashed, current_refs = analyze_new_unconsolidated(length, unique_matrices, unique_matrices_hashed, previous_matrices, previous_matrices_hashed, previous_refs)
        print(time.perf_counter()-t1)
    consolidated_matrices, consolidated_matrices_hashed, consolidated_refs = consolidate_internal(current_matrices, current_matrices_hashed, current_refs)
    
    n_old_matrices = unique_matrices.shape[0]  # We'll just be searching through unique_matrices* for previous occurrences of matrices, so no need to save refs or keep anything unconsolidated
    n_new_matrices = consolidated_matrices.shape[0]
    unique_matrices.resize((n_old_matrices+n_new_matrices, 3, 3))
    unique_matrices[n_old_matrices:, :, :] = consolidated_matrices
    unique_matrices_hashed.resize((n_old_matrices+n_new_matrices))
    unique_matrices_hashed[n_old_matrices:] = consolidated_matrices_hashed
    
    unique_matrices_length.append(current_matrices)  # For unique_*_length* we want to be able to iterate per ref, so we keep things unconsolidated and save the refs
    unique_matrices_length_hashed.append(current_matrices_hashed)
    unique_refs_length.append(current_refs)

    formatted_results = [{"mat": result[0], "refs": result[1]} for result in zip(consolidated_matrices, consolidated_refs)]
    unique_results_all.extend(formatted_results)
    unique_results_length.append(formatted_results)

@numba.jit(nopython=True)
def analyze_new_unconsolidated(length, unique_matrices, unique_matrices_hashed, previous_matrices, previous_matrices_hashed, previous_refs):
    n_previous = previous_matrices.shape[0]
    n_chunks = math.ceil(n_previous/chunk_size)
    chunked_matrices = numba.typed.List()
    chunked_matrices_hashed = numba.typed.List()
    chunked_refs = numba.typed.List()
    for i in range(n_chunks):
        chunked_matrices.append(O3_1.copy())
        chunked_matrices_hashed.append(O1.copy())
        chunked_refs.append(O1.copy())
    
    for i_chunk in range(n_chunks):  # TODO: parallelize by changing range to prange and editing the Numba annotation
        start = i_chunk*chunk_size
        stop = min((i_chunk+1)*chunk_size, n_previous)
        these_previous_matrices = previous_matrices[start:stop]
        these_previous_matrices_hashed = previous_matrices_hashed[start:stop]
        these_previous_refs = previous_refs[start:stop]
        these_matrices, these_matrices_hashed, these_refs = analyze_chunk(unique_matrices, unique_matrices_hashed, these_previous_matrices, these_previous_matrices_hashed, these_previous_refs)
        chunked_matrices[i_chunk] = these_matrices
        chunked_matrices_hashed[i_chunk] = these_matrices_hashed
        chunked_refs[i_chunk] = these_refs

    n_current = 0
    for chunk in chunked_refs:
        n_current += len(chunk)
    current_matrices = np.zeros((n_current, 3, 3), dtype=dtype)
    current_matrices_hashed = np.zeros((n_current), dtype=dtype)
    current_refs = np.zeros((n_current), dtype=dtype)

    for i in range(n_current):
        i_chunk = i // chunk_size
        i_within = i % chunk_size
        current_matrices[i, :, :] = chunked_matrices[i_chunk][i_within, :, :]
        current_matrices_hashed[i] = chunked_matrices_hashed[i_chunk][i_within]
        current_refs[i] = chunked_refs[i_chunk][i_within]

    return current_matrices, current_matrices_hashed, current_refs

@numba.jit(nopython=True)
def analyze_chunk(unique_matrices, unique_matrices_hashed, these_previous_matrices, these_previous_matrices_hashed, these_previous_refs):  # The new heart of the algorithm
    n_previous = these_previous_matrices.shape[0]
    possible_matrices = np.zeros((n_previous*2, 3, 3), dtype=dtype)
    possible_matrices_hashed = np.zeros((n_previous*2), dtype=dtype)
    possible_refs = np.zeros((n_previous*2), dtype=dtype)

    for i_previous in range(n_previous):
        mat_previous = these_previous_matrices[i_previous, :, :]
        mat_A = multiply_3x3(mat_previous, A)
        hash_A = hash_3x3(mat_A)
        mat_B = multiply_3x3(mat_previous, B)
        hash_B = hash_3x3(mat_B)
        if unique_with_hash(mat_A, unique_matrices, hash_A, unique_matrices_hashed):
            i_A = i_previous*2
            ref_A = multiply_ref_A(these_previous_refs[i_previous])
            possible_matrices[i_A, :, :] = mat_A
            possible_matrices_hashed[i_A] = hash_A
            possible_refs[i_A] = ref_A
            i_A = i_previous*2
            ref_A = multiply_ref_A(these_previous_refs[i_previous])
            possible_matrices[i_A, :, :] = mat_A
            possible_matrices_hashed[i_A] = hash_A
            possible_refs[i_A] = ref_A
        if unique_with_hash(mat_B, unique_matrices, hash_B, unique_matrices_hashed):
            i_B = i_previous*2+1
            ref_B = multiply_ref_B(these_previous_refs[i_previous])
            possible_matrices[i_B, :, :] = mat_B
            possible_matrices_hashed[i_B] = hash_B
            possible_refs[i_B] = ref_B

    n_found = np.count_nonzero(possible_refs)
    these_matrices = np.zeros((n_found, 3, 3), dtype=dtype)
    these_matrices_hashed = np.zeros((n_found), dtype=dtype)
    these_refs = np.zeros((n_found), dtype=dtype)
    i_found = 0
    for i_current in range(n_previous*2):
        if possible_refs[i_current] > 0:
            these_matrices[i_found, :, :] = possible_matrices[i_current, :, :]
            these_matrices_hashed[i_found] = possible_matrices_hashed[i_current]
            these_refs[i_found] = possible_refs[i_current]
            i_found += 1
    return these_matrices, these_matrices_hashed, these_refs

@numba.jit(nopython=True)
def consolidate_internal(current_matrices, current_matrices_hashed, current_refs): 
    refs = numba.typed.List()
    for ref in current_refs:
        refs.append(list_with_element(ref))
    n_consolidated = len(current_refs)
    for i in range(len(current_refs)):
        n = find_first_equal_hash(current_matrices[i], current_matrices, current_matrices_hashed[i], current_matrices_hashed)
        if n < i:
            refs[n].append(refs[i][0])
            refs[i] = list_with_element(-1)  # Mark this ref as invalid because it's been moved
            n_consolidated -= 1
    consolidated_matrices = np.zeros((n_consolidated, 3, 3), dtype=dtype)
    consolidated_matrices_hashed = np.zeros((n_consolidated), dtype=dtype)
    consolidated_refs = numba.typed.List()
    for i in range(n_consolidated):
        consolidated_refs.append(list_with_element(0))
    i_consolidated = 0
    for i_current in range(len(current_refs)):
        if refs[i_current][0] > 0:
            consolidated_matrices[i_consolidated, :, :] = current_matrices[i_current, :, :]
            consolidated_matrices_hashed[i_consolidated] = current_matrices_hashed[i_current]
            consolidated_refs[i_consolidated] = refs[i_current]
            i_consolidated += 1
    return consolidated_matrices, consolidated_matrices_hashed, consolidated_refs

def analyze(length):  # Analyze words of length length, assuming analyze(length-1) has already been called for all shorter lengths but not this one
    t1 = time.perf_counter()
    these_unique_matrices, these_unique_refs = analyze_refs(length_to_ref(length), length_to_ref(length+1), unique_matrices, unique_matrices_hashed)
    these_unique_results = [{"mat": result[0], "refs": result[1]} for result in zip(these_unique_matrices, these_unique_refs)]
    unique_results_length.append(these_unique_results)  # Save the per-length results globally (no need to keep the streamlined matrices per length)
    print(time.perf_counter()-t1)
    t1 = time.perf_counter()
    analyze_unique_results(these_unique_results)
    print(time.perf_counter()-t1)

@numba.jit(nopython=True)
def analyze_refs(start, stop, unique_matrices, unique_matrices_hashed):
    these_unique_refs = numba.typed.List()
    these_unique_refs.append(list_with_element(0))
    these_unique_matrices = np.zeros((1,3,3), dtype=dtype)  # A list of all the matrices in these_unique_results, for faster computation. We initialize with an entry because Numba doesn't like empty arrays.
    these_unique_matrices_hashed = np.array(list_with_element(0))  # A list of all the hashes of these_unique_matrices, for even faster computation
    warnings = 0
    for ref in range(start, stop):  # Populate these_unique_results and these_unique_matrices_hashed
        mat, warn = int_to_mat_cached(words_cache, ref, mod=modulo)
        if warn and warnings < max_warnings:
            print("WARNING")
            print(mat)
            warnings += 1

        this_hash = hash_3x3(mat)  # Get a hash for streamlined comparison
        if only_reduced and find_first_equal_hash(mat, unique_matrices, this_hash, unique_matrices_hashed) > 0: continue
        i = find_first_equal_hash(mat, these_unique_matrices, this_hash, these_unique_matrices_hashed)  # We now compare to what we have in these_unique_results so far. We do not subtract one because we drop these_unique_refs's element 0 also 
        if i >= 0:  # If we found a match, we add a reference to this one and move on
            if max_refs_per_length < 0 or len(these_unique_refs[i]) < max_refs_per_length: these_unique_refs[i].append(ref)
        else:  # If we didn't find it, we create a new entry in our results and also add it to the streamlined data structure
            these_unique_refs.append([ref])
            these_unique_matrices = append_numba_3x3(these_unique_matrices, mat)  # I think this is inefficient, but I don't see a good alternative (and it's not done with *that* much data)
            these_unique_matrices_hashed = np.concatenate((these_unique_matrices_hashed, np.array([this_hash])))
    return these_unique_matrices[1:], these_unique_refs[1:]

def analyze_unique_results(unique_results):
    for this_result in unique_results:  # Now we search the results for shorter lengths and look for matches there
        this_hash = hash_3x3(this_result["mat"])
        i = find_first_equal_hash(this_result["mat"], unique_matrices, this_hash, unique_matrices_hashed)-1  # We require that unique_matrices_all contain all unique matrices with reduced forms of shorter length.  Subtract one because we added element 0 manually.
        if i >= 0:
            unique_results_all[i]["refs"].extend(this_result["refs"])  # We add the refs of the current length to the list of existing refs (OK that we don't keep this separated by length because it's easy to test a ref for length)
        else:
            unique_results_all.append({"mat": this_result["mat"], "refs": copy.copy(this_result["refs"])})  # If this_result["refs"] is not copied here, it introduces an annoying bug
            append_inplace_3x3(unique_matrices, this_result["mat"])
            append_inplace_1d(unique_matrices_hashed, this_hash)

def summarize(length):  # Computes summary statistics for words of length length and all words up to length length. Prepare for lots of dense one-liners.
    t1 = time.perf_counter()
    these_unique_results = unique_results_length[length]
    total_words_length = length_to_ref(length)
    canonically_reduced_words_length = count_true([ref_to_length(this_result["refs"][0]) == length for this_result in unique_results_all])  # A word of length length is canonically reduced if it is the first ref for its result
    reduced_words_length = sum([
        0 if ref_to_length(this_result["refs"][0]) != length else
        count_true([
            ref_to_length(ref) == length
            for ref in this_result["refs"]])
        for this_result in unique_results_all])  # A word of length length is reduced if the canonically reduced form of its matrix is of length length
    if not only_reduced: same_length_unique_length = count_true([len(this_result["refs"]) == 1 for this_result in these_unique_results])
    same_and_shorter_unique_length = count_true([
        False if ref_to_length(this_result["refs"][0]) != length else
        count_true([
            ref_to_length(ref) == length
            for ref in this_result["refs"]]) == 1
        for this_result in unique_results_all])  # A word of length length is same-and-shorter unique if it is of length length and is canonically reduced and there are no other reduced forms of its matrix
    reduced_nonunique_length = sum([
        (lambda n: 0 if n == 1 else n)
            (0 if ref_to_length(this_result["refs"][0]) != length else
            count_true([
                ref_to_length(ref) == length
                for ref in this_result["refs"]]))
        for this_result in unique_results_all])  # A word is reduced nonunique if it has the same length as the canonically reduced form of its matrix, if that is length length, and it is not the only one to satisfy this property
    
    totals["total_words"] += total_words_length
    totals["reduced_words"] += reduced_words_length
    totals["canonically_reduced_words"] += canonically_reduced_words_length
    if not only_reduced: totals["same_length_unique"] += same_length_unique_length
    totals["same_and_shorter_unique"] += same_and_shorter_unique_length
    totals["reduced_nonunique"] += reduced_nonunique_length

    states["length_calculated"] = length
    t2 = time.perf_counter()
    timers["summarization_time"] += t2-t1
    log(format_ratio("Summarization time (length="+str(length)+")", t2-t1, timers["summarization_time"]), "summarization_time", print_times_length)
    if print_times_length: print()
    
    log("Length "+str(length)+":", "results_summary")
    log(format_ratio("Total", total_words_length, totals["total_words"]), "results_summary")
    log(format_ratio("Reduced", reduced_words_length, totals["reduced_words"]), "results_summary")
    log(format_ratio("Canonically reduced", canonically_reduced_words_length, totals["canonically_reduced_words"]), "results_summary")
    log("Same-length unique: not computed" if only_reduced else format_ratio("Same-length unique", same_length_unique_length, totals["same_length_unique"]), "results_summary")
    log(format_ratio("Same-and-shorter unique", same_and_shorter_unique_length, totals["same_and_shorter_unique"]), "results_summary")
    log(format_ratio("Reduced nonunique", reduced_nonunique_length, totals["reduced_nonunique"]), "results_summary")
    log("", "results_summary")

def save_output(force=False):
    if not force:
        if not write_files: return  # Stop if we're not supposed to write files
        if states["length_saved"] >= states["length_calculated"]: return  # Stop if there is no new information
    
    # Put together the text output
    output_log = "length_calculated="+str(states["length_calculated"])+"\n"
    output_log += "modulo=" + (str(modulo) if modulo > 1 else "NONE")
    output_log += "\n\n"
    output_log += "\n\n".join(["LOG "+log_string_key+":\n"+log_strings[log_string_key] for log_string_key in log_strings])
    output_results_all = results_to_string(unique_results_all, "all")
    output_results_length = "\n\n".join([results_to_string(unique_result_length, "length="+str(i)) for i, unique_result_length in enumerate(unique_results_length)])

    output_dir = output_prefix+("_mod_"+str(modulo) if modulo > 1 else "")
    # Create the output folder if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the text output
    write_output(output_log, output_dir, "output_log.txt")
    write_output(output_results_all, output_dir, "output_results_all.txt")
    write_output(output_results_length, output_dir, "output_results_length.txt")

    # Pickle the important variables so they can be further analyzed programmatically
    pickle_data(unique_results_all, output_dir, "results_all.pickle")
    pickle_data(unique_results_length, output_dir, "results_length.pickle")

    states["length_saved"] = states["length_calculated"]

def write_output(output, output_dir, filename):
    with open(os.path.join(output_dir, filename), "w") as output_file:
        output_file.write(output)

def pickle_data(data, output_dir, filename):
    with open(os.path.join(output_dir, filename), "wb") as data_file:
        pickle.dump(data, data_file)

def results_to_string(results, label):
    matrices = "MATRICES ("+label+"):\n"
    refs = "REFS ("+label+"):\n"
    for i, result in enumerate(results):
        matrices += str(i+1)+" (CRR="+str(result["refs"][0])+"):\n"+str(result["mat"])+"\n"  # CRR: canonical reduced ref(erence)
        refs += str(i+1)+": "+str(result["refs"])+"\n"
    return matrices+"\n"+refs

def format_ratio(label, num, denom):  # Nicely formats a label and a ratio
    return label+": "+str(num)+"/"+str(denom)

def log(output, log_string, print_output=True):
    if print_output: print(output)
    log_strings[log_string] += output+"\n"

@numba.jit(numba.types.Tuple((numba_dtype[:,:], numba.types.bool_))(numba.types.Array(numba_dtype, 3, "A", readonly=True), numba_dtype, numba_dtype), nopython=True)
def int_to_mat_cached(words_cache, ref, mod=0):
    ref1 = ref_first_n(ref, cache_length)
    ref2 = ref_last_n(ref, cache_length)
    result = multiply_3x3(words_cache[ref1,:,:], words_cache[ref2,:,:])
    warn = warn_3x3(result)
    if mod > 1: result %= mod
    return result, warn

@numba.jit(numba_dtype(numba.types.Array(numba_dtype, 2, "A", readonly=True), numba.types.Array(numba_dtype, 3, "A", readonly=True), numba_dtype, numba.types.Array(numba_dtype, 1, "A", readonly=True)), nopython=True)  # In this case, if we don't manually specify type, Numba actually slows things down.
def find_first_equal_hash(elem, elem_list, elem_hash, elem_hash_list):  # Returns the index of the first value of elem_list equal to elem, or -1 if there is no match
    for i in range((len(elem_list))):
        if elem_hash == elem_hash_list[i] and equals_lazy_3x3(elem, elem_list[i,:,:]): return i
    return -1

@numba.jit(numba.types.bool_(numba.types.Array(numba_dtype, 2, "C", readonly=True), numba.types.Array(numba_dtype, 3, "C", readonly=True), numba_hashtype, numba.types.Array(numba_hashtype, 1, "C", readonly=True)), nopython=True)
def unique_with_hash(elem, elem_list, elem_hash, elem_hash_list):  # This is the part that really needs to be optimized
    i = 0
    s = 8
    l = len(elem_list)-s
    while i < l:  # Unroll for a little more speed (TODO: be more rigorous about this)
        if elem_hash == elem_hash_list[i] and equals_lazy_3x3(elem, elem_list[i]): return False
        if elem_hash == elem_hash_list[i+1] and equals_lazy_3x3(elem, elem_list[i+1]): return False
        if elem_hash == elem_hash_list[i+2] and equals_lazy_3x3(elem, elem_list[i+2]): return False
        if elem_hash == elem_hash_list[i+3] and equals_lazy_3x3(elem, elem_list[i+3]): return False
        if elem_hash == elem_hash_list[i+4] and equals_lazy_3x3(elem, elem_list[i+4]): return False
        if elem_hash == elem_hash_list[i+5] and equals_lazy_3x3(elem, elem_list[i+5]): return False
        if elem_hash == elem_hash_list[i+6] and equals_lazy_3x3(elem, elem_list[i+6]): return False
        if elem_hash == elem_hash_list[i+7] and equals_lazy_3x3(elem, elem_list[i+7]): return False
        i += s
    for i in range(i, len(elem_list)):
        if elem_hash == elem_hash_list[i] and equals_lazy_3x3(elem, elem_list[i]): return False
    return True

@numba.jit(nopython=True)
def append_numba_3x3(mat, to_append):
    result = np.concatenate((mat, O3_1))
    result[result.shape[0]-1, :, :] = to_append
    return result

def append_inplace_1d(mat, to_append):  # Not Numba safe, though could be made so
    mat.resize((mat.shape[0]+1), refcheck=False)
    mat[mat.shape[0]-1] = to_append

def append_inplace_3x3(mat, to_append):  # Not Numba safe, though could be made so
    mat.resize((mat.shape[0]+1, 3, 3), refcheck=False)
    mat[mat.shape[0]-1, :, :] = to_append

if __name__ == "__main__":
    main()