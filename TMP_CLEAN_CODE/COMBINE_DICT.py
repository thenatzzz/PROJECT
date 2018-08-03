from RANDOM_TOPOLOGY import save_topology, fix_dict
from QLEARNING import open_file
from CHECK_MODEL_DICT import format_data_without_header
from operator import itemgetter
import os


def list_file_in_path(path):
    array_file_name = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return array_file_name

def add_path_to_file(path,file):
    file_with_path = path + "/"+file
    return file_with_path

def combine(path,array_file):
    final_list_data = []
    for file in array_file:
        file_with_path = add_path_to_file(path, file)
        data = open_file(file_with_path)
        data = format_data_without_header(data)
        final_list_data += data
    return final_list_data

def fix_list(data_list):
    final_array = []
    num_model_fixed = 0
    for array in data_list:
        temp_array = array[:]
        array = array[1:-2]
        count = 0
        for single_element in array:
            if single_element != '-':
                count += 1
        if count < 4 and array[count-1] != 's':
            # print("Before fixed: ",array)
            array[count] = 's'
            temp_array[count+1] = 's'
            num_model_fixed += 1
            # print("After fixed: ",array)
        final_array.append(temp_array)
    # print(num_model_fixed)
    return final_array

def check_dup(data, final_list):
    model_name = data[0]
    for data_in_final in final_list:
        model_name_final_list = data_in_final[0]
        if model_name == model_name_final_list:
            return True
    return False

def delete_duplicate(data_list):
    final_list = []
    num = 0
    for data in data_list:
        is_duplicate = check_dup(data,final_list)

        if is_duplicate:
            continue
        else:
            final_list.append(data)

    return final_list

def strip_model_tag(data):
    temp_model_num = data[0].strip('model_')
    return int(temp_model_num)

def strip_model_list(data_list):
    model_index = 0
    for data in data_list:
        data[model_index] = strip_model_tag(data)
    print(data_list)
    return data_list

def sort_data(data_list, index_to_sort):
    data_list = sorted(data_list, key = itemgetter(index_to_sort))
    return data_list

def sort_data_by_name(data_list):
    data_list = strip_model_list(data_list)
    model_name_index = 0
    data_list = sort_data(data_list, model_name_index )
    return data_list

def attach_model_tag(data_list):
    for data in data_list:
        data[0] = "model_"+str(data[0])
    return data_list

def fix_layer(data_list):
    LETTER_INDEX  =0
    for data in data_list:
        for index in range(1,5):
            data[index] = data[index][LETTER_INDEX].lower()+data[index][LETTER_INDEX+1:]
    return data_list

def clean_model_dict(data_list):
    modified_data = fix_list(data_list)
    modified_data = delete_duplicate(modified_data)
    modified_data = sort_data_by_name(modified_data)
    modified_data = attach_model_tag(modified_data)
    modified_data = fix_layer(modified_data)
    # modified_data = sort_data(modified_data, -2)

    # print(modified_data)
    return modified_data

def combine_file(path,output_file):
    array_files = list_file_in_path(path)
    # print(array_files)
    data = combine(path,array_files)
    # print("Before clean-> length data: ", len(data))
    data = clean_model_dict(data)
    # print(data)
    # print("After clean-> length_data: ",len(data))
    # print(data)
    filename = save_topology(output_file,data)

path = "/homes/nj2217/PROJECT/TMP_CLEAN_CODE/CIFAR_DICT"
output_file = "/homes/nj2217/PROJECT/TMP_CLEAN_CODE/CIFAR_DICT/COMPLETE_CIFAR10.csv"
combine_file(path,output_file)
