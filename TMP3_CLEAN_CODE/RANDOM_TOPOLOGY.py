import random
import csv
import os
import pandas as pd

'''
Convolution layers: num.output_filter, kernel_size, stride
c_1 = 32,3,1
c_2 = 32,4,1
c_3 = 32,5,1
c_4 = 36,3,1
c_5 = 36,4,1
c_6 = 36,5,1
c_7 = 48,3,1
c_8 = 48,4,1
c_9 = 48,5,1
c_10 = 64,3,1
c_11 = 64,4,1
c_12 = 64,5,1

Max pooling layers: kernel , stride
m_1 = 2,2
m_2 = 3,2
m_3 = 5,3

Softmax
s
'''

dict_element = {}
dict_element['c_1'] = [32,3,1]
dict_element['c_2'] = [32,4,1]
dict_element['c_3'] = [32,5,1]
dict_element['c_4'] = [36,3,1]
dict_element['c_5'] = [36,4,1]
dict_element['c_6'] = [36,5,1]
dict_element['c_7'] = [48,3,1]
dict_element['c_8'] = [48,4,1]
dict_element['c_9'] = [48,5,1]
dict_element['c_10'] = [64,3,1]
dict_element['c_11'] = [64,4,1]
dict_element['c_12'] = [64,5,1]
dict_element['m_1'] = [2,2]
dict_element['m_2'] = [3,2]
dict_element['m_3'] = [5,3]
dict_element['s'] = [1]

MAX_NUM_LAYER = 4

def have_duplicate(dict_model,temp_list):
    for key,value in dict_model.items():
        if temp_list == value:
            return True
    return False

def count_non_duplicate(dict_model):
    temp_list = []
    for key,value in dict_model.items():
        temp_value = ''.join(value)
        temp_list.append(temp_value)
    return len(set(temp_list))

def fix_topology(dict_model):
    list_of_dict = []
    for key,value in dict_model.items():
        temp_dict = {}
        temp_dict['Model'] = key

        temp_dict['1st Layer'] = value[0]
        if len(value) < 2:
            temp_dict['2nd Layer'] = '-'
            temp_dict['3rd Layer'] = '-'
            temp_dict['4th Layer'] = '-'
            list_of_dict.append(temp_dict)
            continue

        temp_dict['2nd Layer'] = value[1]
        if len(value) < 3:
            temp_dict['3rd Layer'] = '-'
            temp_dict['4th Layer'] = '-'
            list_of_dict.append(temp_dict)
            continue

        temp_dict['3rd Layer'] = value[2]
        if len(value) < 4:
            temp_dict['4th Layer'] = '-'
            list_of_dict.append(temp_dict)
            continue

        temp_dict['4th Layer'] = value[3]

        list_of_dict.append(temp_dict)
    return list_of_dict

def add_layer(number_model):
    num_model = 0
    dict_model = {}

    while num_model < number_model:
        temp_list = []
        for num_layer in range(MAX_NUM_LAYER):

            element = random.choice(list(dict_element))

            if num_layer == 0:
                if element == 's' or element == 'm_1' or element == 'm_2' or element == 'm_3' :
                    continue
            if num_layer == 1:
                if element == 's':
                    continue
            temp_list.append(element)

            if element == 's':
                break
        if len(temp_list) < 4 and temp_list[-1] != 's':
            temp_list.append(random.choice(list(dict_element)))

        if have_duplicate(dict_model,temp_list):
            continue

        dict_model["model_"+str(num_model+1)] = temp_list
        num_model += 1
    return dict_model

def implement_topology(number_model):
    dict_model = add_layer(number_model)

    # TO CHECK FOR NON-DUPLICATE: use function below
    num_non_dup = count_non_duplicate(dict_model)
    list_of_dict = fix_topology(dict_model)

    return list_of_dict

# Put all models into csv file
def save_topology(file_name,data_list):
    list_of_data = data_list[:]
    csv_columns = [['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy','Loss']]
    data_list = csv_columns + data_list

    myFile = open(file_name,'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data_list)
    return file_name
    
# TO ADD: LAST COLUMN OF FILE, use this function
def fn_to_add_column(file_name, content, column_name):
    csv_input = pd.read_csv(file_name)
    csv_input[column_name] =  content
    csv_input.to_csv(file_name, index= False)

def convert_dict_to_list(dict_data):
    list_of_model = []
    for indi_list in dict_data:
        temp_list = [indi_list['Model'],indi_list['1st Layer'], indi_list['2nd Layer'],
                    indi_list['3rd Layer'], indi_list['4th Layer'],'Unknown','Unknown']
        list_of_model.append(temp_list)
    return list_of_model

def fix_dict(dict_data):
    final_array = []
    data  = convert_dict_to_list(dict_data)
    for array in data:
        temp_array = array[:]
        array = array[1:-2]
        count = 0
        for single_element in array:
            if single_element != '-':
                count += 1
        if count < 4 and array[count-1] != 's':
            array[count] = 's'
            temp_array[count+1] = 's'
        final_array.append(temp_array)
    return final_array

def get_random_topology(num_model, output_file_name):
    data_dict = implement_topology(num_model)
    data_dict = fix_dict(data_dict)
    output_file_name= save_topology(output_file_name,data_dict)

    return output_file_name
