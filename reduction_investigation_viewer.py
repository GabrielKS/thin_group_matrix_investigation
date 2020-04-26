import os
import pickle

max_length = None
unique_results_all = None
unique_results_length = None

def main():
    load("output")
    print(max_length)
    # print(unique_results_all)
    # print(unique_results_length)

def load(output_dir):
    global max_length, unique_results_all, unique_results_length
    with open(os.path.join(output_dir, "output_log.txt")) as output_log:
        max_length = int(output_log.readline().split("=")[1])
    unique_results_all = unpickle_data(os.path.join(output_dir, "results_all.pickle"))
    unique_results_length = unpickle_data(os.path.join(output_dir, "results_length.pickle"))

def unpickle_data(filename):
    data = None
    with open(filename, "rb") as data_file:
        data = pickle.load(data_file)
    return data

if __name__ == "__main__":
    main()