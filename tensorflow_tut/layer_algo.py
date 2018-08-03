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

MAX_NUM_MODEL = 1000
MAX_NUM_LAYER = 4

def have_duplicate(dict_model,temp_list):
    for key,value in dict_model.items():
        if temp_list == value:
            return True
    return False

dict_model = {}
num_model = 1500
temp_num_model = num_model
# while num_model < MAX_NUM_MODEL:
while num_model < temp_num_model+MAX_NUM_MODEL:

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

    # print(temp_list)

    # FOR 2 last columns: Accuracy and Loss
    # temp_list.append('Unknown')
    # temp_list.append('Unknown')

    if have_duplicate(dict_model,temp_list):
        continue

    dict_model["model_"+str(num_model+1)] = temp_list
    num_model += 1

# TO CHECK FOR NON-DUPLICATE: use function below
temp_list = []
for key,value in dict_model.items():
    temp_value = ''.join(value)
    temp_list.append(temp_value)
print("NUMBER OF non-duplicate: ",len(set(temp_list)))


# csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer']
list_of_dict = []

for key,value in dict_model.items():
    temp_dict = {}
    temp_dict['Model'] = key
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: ",temp_dict, "; key: ",key)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: ",temp_dict, "; value: ",value)

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

    # 2 last columns for Accuracy and Loss
    # temp_dict['Accuracy'] = value[4]
    # temp_dict['Loss'] = value[5]
    # print("temp_dict:  ", temp_dict)
    # print("key : ", key)
    # print("~~~~~~~~~~~~~~~~~~~LENGTH OF temp_dict: ", len(temp_dict))
    # print('\n')

    list_of_dict.append(temp_dict)


# Put all models into csv file
csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer']
# csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer', 'Accuracy','Loss']
# csv_file = "Names_no_dup.csv"
csv_file = "Model1501-2500.csv"

# csv_file = "Names_no_dup_2.csv"

try:
    with open(csv_file, 'w') as csvfile:
    # with open(csv_file, 'a') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        for data in list_of_dict:
            writer.writerow(data)
    print("Create csv file named ", csv_file, " successfully!!")
except IOError:
    print("I/O error")

# TO ADD: LAST COLUMN OF FILE, use this function
def fn_to_add_column(file_name, content, column_name):
    csv_input = pd.read_csv(file_name)
    csv_input[column_name] =  content
    csv_input.to_csv(file_name, index= False)

fn_to_add_column(csv_file,'Unknown','Accuracy')
fn_to_add_column(csv_file,'Unknown','Loss')


'''
Count how many row in csv file
filename = '....csv'
n= sum(1 for line in open(filename))
'''

'''
Read csv file in dictionary format
with open("Names.csv") as f:
    reader = csv.DictReader(f)
    data = [r for r in reader]
'''

'''
# Read csv file in list format
# with open('Names.csv') as f:
with open('Names_no_dup.csv') as f:

    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data1  = [r for r in reader]
'''

'''
# To get index of models that have not been trained
index = 0
while data1[index][-1] != "":
    index += 1
row_at_empty_cell = index
content_at_empty_cell = data1[index]
print("Content at empty cell: " ,content_at_empty_cell)
print("Index at empty cell: ", row_at_empty_cell)
'''

'''
TEST Global Index

global_index = 0
sample_dict = {'model_1': ['c_6', 'c_6', 'c_3', 'c_1'], 'model_2': ['c_11', 'c_1', 'c_6', 'c_11'], 'model_3': ['m_2', 'm_3', 'c_8'], 'model_4': ['c_2', 'c_2', 'c_10', 'c_11'], 'model_5': ['c_11', 'c_11', 'c_5', 'm_1'], 'model_6': ['c_2', 'c_12', 'c_4', 'c_10'], 'model_7': ['c_12', 'm_1', 's'], 'model_8': ['c_5', 'm_1', 'c_11', 'm_3'], 'model_9': ['c_4', 'c_6', 'c_11', 'c_10'], 'model_10': ['c_9', 'c_7', 's']}

def main():
    global global_index
    while global_index < len(sample_dict):
        print(global_index)
        global_index += 1
main()
'''
