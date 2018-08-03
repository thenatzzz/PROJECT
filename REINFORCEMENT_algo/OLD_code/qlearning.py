import numpy as np
import random
import csv
import pandas

MAX_EPISODE = 1
MIN_ALPHA = 0.1
# eps = 0.2
eps = 0.8

gamma = 1.0
alpha_list = np.linspace(1.0, MIN_ALPHA, MAX_EPISODE)

layer = { 'c_1':0, 'c_2':0, 'c_3':0, 'c_4':0, 'c_5':0, 'c_6':0, \
        'c_7':0, 'c_8':0, 'c_9':0, 'c_10':0, 'c_11':0, 'c_12':0,\
        'm_1':0,'m_2':0,'m_3':0,'s':0}


q_table = []
for i in range(4):
    q_table.append(layer)
q_table = np.asarray(q_table)

file_path = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/"
file_name = "model.csv"
file_csv = file_path + file_name
with open(file_csv) as f:
    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data  = [r for r in reader]
print("data: ",data)
print('\n\n')

def choose_action(num_layer):
    if random.uniform(0,1) < eps:
        # get random keys of q_table[layer]
        random_key, random_value = random.choice(list(q_table[num_layer].items()))
        return random_key,random_value
    else:
        max_key = max(q_table[num_layer], key= q_table[num_layer].get)
        max_value = q_table[num_layer][max_key]
        return max_key,max_value

def fn_format_action_array(action_array):
    new_action_array = action_array[:]
    length_action_array = len(action_array)
    if length_action_array != 4:
        for i in range(4-length_action_array):
            new_action_array.append('-')
    return new_action_array

def train_model(action_array):
    return 0.5
def get_accuracy(action_array):
    new_action_array = action_array[:]
    new_action_array = fn_format_action_array(new_action_array)
    for index in range(len(data)):

        if np.array_equal(new_action_array, data[index][1:-2]):
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
            print("+++++++++ THERE IS A MATCH !! +++++++++++++++++")
            print("at model number: ", data[index][0])
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
            print("Accuray of model: ",data[index][-2])
            print("''''''''''''''''''''''''''''''",q_table)

            return float(data[index][-2])
    print(action_array)
    return train_model(action_array)

def get_next_value(num_layer,action_array):
    '''set num_layer to next layer '''
    num_layer += 1
    if num_layer == 4 or action_array[-1] == 's':
        # print("~~~~~~~~~~~~~~~~",q_table)
        return get_accuracy(action_array)
    else:
        max_key = max(q_table[num_layer], key = q_table[num_layer].get)
        max_value = q_table[num_layer][max_key]
        # print("#################################",q_table)
        return max_value

def round_value(temp_q_table):
    for k, v in temp_q_table.items():
        temp_q_table[k]= float('%.5f' % round(v,5))
    return temp_q_table

for i in range(MAX_EPISODE):
    action = 'non'
    num_layer = 0
    alpha = alpha_list[i]
    print("alpha: ", alpha)
    action_array = []
    while action != 's' and num_layer < 4:
        # print(q_table)
        action,value_action = choose_action(num_layer)
        # print("action: ",action, " + value_action: ",value_action)
        action_array.append(action)
        max_value_next_action = get_next_value(num_layer, action_array)
        # q_table[num_layer][action] += 3
        # print(q_table)

        # q_table[num_layer][action] = q_table[num_layer][action] + \
                    # alpha*(gamma*max_value_next_action-q_table[num_layer][action])
        td_target = gamma *max_value_next_action
        # print("---------------- 1 ----------------------- ",q_table)
        # print("max_value for next_action: ",max_value_next_action)
        td_delta = td_target - q_table[num_layer][action]
        # print("td_delta: ",td_delta)
        # print("---------------- 2------------------------ ",q_table)
        # print(alpha)

        # q_table[num_layer][action] += td_delta
        # print("q_table[num_layer][action] = ",q_table[num_layer][action])

        q_table[num_layer][action] += alpha * td_delta
        # print("---------------- 3------------------------ ",q_table)
        q_table[num_layer] = round_value(q_table[num_layer])
        print(" num_layer : ",num_layer)
        # print("q_table at layer ",num_layer," is : ",q_table)
        print('\n')
        num_layer += 1

    # q_table[num_layer-1][action] += 3
    print("number of episode: ", i)
    print(action_array)
    print('\n')
# q_table[num_layer-1][action] += 3

print('----------------END------------------------------')
print('\n')
print(q_table)
print('\n')
print('\n')
# print(q_table[0])
# print(q_table[1])


layer2 = { 'c_1':0, 'c_2':0, 'c_3':0, 'c_4':0, 'c_5':0, 'c_6':0, \
        'c_7':0, 'c_8':0, 'c_9':0, 'c_10':0, 'c_11':0, 'c_12':0,\
        'm_1':0,'m_2':0,'m_3':0,'s':0}
layer2_2 = { 'c_1':0, 'c_2':0, 'c_3':0, 'c_4':0, 'c_5':0, 'c_6':0, \
        'c_7':0, 'c_8':0, 'c_9':0, 'c_10':0, 'c_11':0, 'c_12':0,\
        'm_1':0,'m_2':0,'m_3':0,'s':0}
layer2_3 = { 'c_1':0, 'c_2':0, 'c_3':0, 'c_4':0, 'c_5':0, 'c_6':0, \
        'c_7':0, 'c_8':0, 'c_9':0, 'c_10':0, 'c_11':0, 'c_12':0,\
        'm_1':0,'m_2':0,'m_3':0,'s':0}
q_table2 = []
q_table2.append(layer2)
q_table2.append(layer2_2)
q_table2.append(layer2_3)
q_table2 = np.asarray(q_table2)
q_table2[2]['c_2'] += q_table2[2]['c_2']+20
print(q_table2)
