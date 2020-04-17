from common import *
import conversion

max_length = 5

words = [None]  # Element 0 of words is None. Element i of words for i>0 is a dictionary containing information about the word corresponding to that index

def main():
    for length in range(max_length+1): populate(length)  # Fast for max_length=10, takes some time for 15, takes a minute or two for 20
    for length in range(max_length+1): analyze(length)
    for word in words: summarize(word)
    # print(words)

def populate(length):  # Populate words with all the words of length length, assuming populate(length-1) has already been called if length>0
    assert len(words) == 2**length
    these_words = [{} for i in range(2**length)]
    for i in range(0, 2**length):
        these_words[i]["int"] = 2**length+i
        these_words[i]["word"] = conversion.int_to_h(these_words[i]["int"])
        these_words[i]["length"] = length
        these_words[i]["mat"] = conversion.h_to_mat(these_words[i]["word"])
    words.extend(these_words)

def analyze(length):  # Analyze words of length length
    these_words = words[2**length:2**(length+1)]
    shorter_words = words[1:2**length]
    for this_word in these_words:
        this_word["equal_same_length"] = []
        for other_word in these_words:
            if other_word["int"] == this_word["int"]: continue
            if np.array_equal(this_word["mat"], other_word["mat"]): this_word["equal_same_length"].append(other_word["int"])

        this_word["equal_shorter_length"] = []
        for other_word in shorter_words:
            if np.array_equal(this_word["mat"], other_word["mat"]): this_word["equal_shorter_length"].append(other_word["int"])

        this_word["reduced"] = (len(this_word["equal_shorter_length"]) == 0)

        equal_shorter_or_same = this_word["equal_shorter_length"]+this_word["equal_same_length"]+[this_word["int"]]  # Note that this is a list of ints, not a list of dicts
        this_word["reduced_length"] = min([other_word for other_word in equal_shorter_or_same])

        this_word["reduced_forms"] = []
        for other_word in equal_shorter_or_same:
            if other_word == this_word["reduced_length"]: this_word["reduced_forms"].append(other_word)

def summarize(word):
    if word is None: return
    print("Word "+str(word["int"])+" ("+word["word"]+") reduced forms:"+str([conversion.int_to_h(s) for s in word["reduced_forms"]]))  # if len(word["reduced_forms"]) != 1: 

if __name__ == "__main__":
    main()