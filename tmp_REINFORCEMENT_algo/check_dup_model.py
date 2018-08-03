import numpy as np
import csv


file_path = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/"
file_name = "model.csv"
file_csv = file_path + file_name
with open(file_csv) as f:
    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data  = [r[1:-1] for r in reader]

# print(data)

num_duplicate = 0
dict_model_dup = {}
for i in range(1, len(data)):
    str_data = str(data[i])
    if str_data in dict_model_dup.keys():
        dict_model_dup[str_data] += 1
    else:
        dict_model_dup[str_data] = 1

list = []

for k,v in dict_model_dup.items():
    if v > 1:
        temp_tuple =(k,v)
        # list.append(k)
        list.append(temp_tuple)

print(dict_model_dup)
print(list)


file_path = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/"
file_name = "model_dict.csv"
file_csv = file_path + file_name
with open(file_csv) as f:
    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data  = [r for r in reader]
last_model = str(data[-1][0])
temp_new_model = last_model.strip('model_')
new_number = int(temp_new_model)+1
new_model = "model_"+str(new_number)
print(new_model)
