from common import *
import conversion
import time
import copy

max_length = 15

words = [None]  # Element 0 of words is None. Element i of words for i>0 is a dictionary containing information about the word corresponding to that index

def main():
    t1 = time.perf_counter()
    for length in range(max_length+1): populate(length)  # Fast for max_length=10, takes some time for 15, takes a minute or two for 20
    t2 = time.perf_counter()
    print("TOTAL population time: "+str(t2-t1)+"\n")

    t1 = time.perf_counter()
    for length in range(max_length+1): analyze(length)
    t2 = time.perf_counter()
    print("TOTAL analysis time: "+str(t2-t1)+"\n")

    for length in range(max_length+1):  summarize(length)
    # print(words)

def populate(length):  # Populate words with all the words of length length, assuming populate(length-1) has already been called for length>0
    t1 = time.perf_counter()

    assert len(words) == 2**length
    these_words = [{} for i in range(2**length)]
    for i in range(0, 2**length):
        these_words[i]["int"] = 2**length+i
        these_words[i]["word"] = conversion.int_to_h(these_words[i]["int"])
        these_words[i]["length"] = length
        these_words[i]["mat"] = conversion.h_to_mat(these_words[i]["word"])
        these_words[i]["analyzed"] = False
    words.extend(these_words)
    
    t2 = time.perf_counter()
    print("Population time (length="+str(length)+"): "+str(t2-t1))

def analyze(length):  # Analyze words of length length, assuming analyze(length-1) has already been called for length>0. If find_all_shorter, this will find all equal shorter words; if not, it will only find a shortest
    t1 = time.perf_counter()
    
    these_words = words[2**length:2**(length+1)]
    shorter_words = words[1:2**length]

    these_words_3d = np.stack([np.asarray(this_word["mat"]) for this_word in these_words])  # Stack the matrices in these_words into a three-dimensional array for faster comparison
    shorter_words_3d = np.stack([np.asarray(this_word["mat"]) for this_word in shorter_words]) if len(shorter_words) > 0 else np.empty((0,3,3))

    for this_word in these_words:
        if this_word["analyzed"]: continue

        this_mat = np.asarray(this_word["mat"])

        this_word["equal_same_length"] = []
        same_length_comparison = np.equal(this_mat, these_words_3d).all(axis=1).all(axis=1)  # Compare this_mat to these_words_3d and "and" all the results together across the dimensions that correspond to the original matrices, so we get a one-dimensional array of boolean values corresponding to whether this_mat is equal to each matrix in these_words
        for i, v in enumerate(same_length_comparison):  # Loop through the boolean array, keeping track of index and value
            if v: this_word["equal_same_length"].append(these_words[i]["int"])  # If we have a match, record the number of the matrix we've matched with

        this_word["equal_shorter_length"] = []
        shorter_length_comparison = np.equal(this_mat, shorter_words_3d).all(axis=1).all(axis=1)  # This method of comparison (here and above) results in roughly 18x faster analysis compared to looping through all the matrices and comparing one at a time
        for i, v in enumerate(shorter_length_comparison):
            if v: this_word["equal_shorter_length"].append(shorter_words[i]["int"])

        this_word["reduced"] = (len(this_word["equal_shorter_length"]) == 0)

        ints_equal_shorter_or_same_length = this_word["equal_shorter_length"]+this_word["equal_same_length"]  # Note that this is a list of ints, not a list of dicts
        this_word["reduced_length"] = min([other_word for other_word in ints_equal_shorter_or_same_length])

        this_word["reduced_forms"] = []
        for other_word in ints_equal_shorter_or_same_length:
            if other_word == this_word["reduced_length"]: this_word["reduced_forms"].append(other_word)

        this_word["unique_shorter_or_same_length"] = (len(ints_equal_shorter_or_same_length) == 1)  # 1 because ints_equal_shorter_or_same_length includes this_word
        this_word["unique_same_length"] = (len(this_word["equal_same_length"]) == 1)
        this_word["reduced_nonunique"] = this_word["reduced"] and not this_word["unique_same_length"]

        this_word["analyzed"] = True

        for other_i in this_word["equal_same_length"]:  # Optimize by copying information for equal words of the same length (makes analysis roughly 8x faster)
            if words[other_i]["analyzed"]: continue
            words[other_i].update(copy.deepcopy(this_word))

    t2 = time.perf_counter()
    print("Analysis time (length="+str(length)+"): "+str(t2-t1))

def summarize(length):
    # t1 = time.perf_counter()

    these_words = words[2**length:2**(length+1)]
    print("Length "+str(length)+":")
    print("Total: "+str(len(these_words)))
    print("Reduced: "+str(np.count_nonzero([this_word["reduced"] for this_word in these_words])))
    print("Same-length unique: "+str(np.count_nonzero([this_word["unique_same_length"] for this_word in these_words])))
    print("Same-and-shorter unique: "+str(np.count_nonzero([this_word["unique_shorter_or_same_length"] for this_word in these_words])))
    print("Reduced nonunique: "+str(np.count_nonzero([this_word["reduced_nonunique"] for this_word in these_words])))
    print()

    # t2 = time.perf_counter()
    # print("Summarization time: "+str(t2-t1))

if __name__ == "__main__":
    main()