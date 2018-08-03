from QLEARNING import open_file
from RANDOM_TOPOLOGY import save_topology
# import random
# import csv
# import os
# import pandas as pd

def get_bad_topology(data,accuracy_threshold):
    list_data = []
    for index in range(len(data)):
        if float(data[index][-2]) < accuracy_threshold:
            list_data.append(data[index])
    return list_data



def main():
    file_name = "fixed_model_dict.csv"
    data = open_file(file_name)
    print(data)
    output_file = "bad_model.csv"
    accuracy_threshold = 0.9
    bad_model = get_bad_topology(data[1:],accuracy_threshold)
    print(bad_model)
    bad_model_file = save_topology(output_file,bad_model)

if __name__ == "__main__":
    main()
