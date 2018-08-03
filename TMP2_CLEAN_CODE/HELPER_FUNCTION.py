import csv

def get_data_from_csv(file_name):
    list_data = []
    with open(file_name, 'rt',encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            list_data.append(row)
    return list_data[:]

def format_data_without_header(data):
    return data[1:]

def get_topology_only(single_model):
    return single_model[1:-2]

def check_complete_model(single_model):
    if len(single_model) == 7:
        return True
    else:
        return False

def count_model_layer(model_from_csv):
    count = 0
    for i in range(len(model_from_csv)):
        count += 1
        if model_from_csv[i] == 's':
            break
    return count

def get_new_model_number(old_model_number):
    return old_model_number+1

def get_current_model_number(latest_model):
    cur_model_num = latest_model.strip('model_')
    return int(cur_model_num)

def get_new_model(lastest_model):
    temp_new_model = get_current_model_number(lastest_model)
    new_number = get_new_model_number(temp_new_model)
    new_model = "model_"+ str(new_number)

    return new_model

def get_latest_model_list(single_model,file):
    file_name = file
    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    lastest_model = data[-1][0]
    new_model = get_new_model(lastest_model)
    new_single_model = [new_model]+single_model+["Unknown","Unknown"]

    return new_single_model
