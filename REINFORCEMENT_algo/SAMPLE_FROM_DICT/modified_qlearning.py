import numpy as np
import random
import csv
import pandas
import os
import time

from modified_train_model import train
# 134,143,168,174,188,201,209,287,393,496,545,577,716,806,994,1056,1174,1264,1275,1452,1453,1458,1937,2003,2129,2139,
# 2149,2067,2073

from enum import Enum
# from train_model2 import train

cont_episode = 0

MAX_ACTION = 16
MAX_STATE = 4
class Action(Enum):
    c_1 = 0
    c_2 = 1
    c_3 = 2
    c_4 = 3
    c_5 = 4
    c_6 = 5
    c_7 = 6
    c_8 = 7
    c_9 = 8
    c_10 = 9
    c_11 = 10
    c_12 = 11
    m_1 = 12
    m_2 = 13
    m_3 = 14
    s = 15
class State:
    layer_1 = 0
    layer_2 = 1
    layer_3 = 2
    layer_4 = 3

''' Create MODEL corresponding to specific epsilon '''
NUM_LIST = 3
NUM_LIST_1 = 1
NUM_LIST_2 = 2
NUM_LIST_3 = 3
''' Change this to specify number of model corresponding to each specific epsilon'''
NUM_MODEL_1 = 1500
NUM_MODEL_2 = 500
# NUM_MODEL_2 = 0

# NUM_MODEL_3 = 150
NUM_MODEL_3 = 500

MAX_EPISODE = NUM_MODEL_1 + NUM_MODEL_2 + NUM_MODEL_3

''' Discount Rate is set to 1.0 as to NOT prioritize any specific layer'''
gamma = 1.0

''' Alpha = learning rate '''
MIN_ALPHA = 0.01
alpha_list = np.linspace(1.0, MIN_ALPHA, MAX_EPISODE) # set alpha at decreasing order
alpha_list = [MIN_ALPHA]*MAX_EPISODE # set alpha equal to MIN_ALPHA

epsilon_dict = {}
epsilon_1 = 1
epsilon_list_2 = np.linspace(0.9,0.7,NUM_MODEL_2)
epsilon_list_3 = np.linspace(0.6,0.1, NUM_MODEL_3)
epsilon_dict[NUM_LIST_1] = epsilon_1
epsilon_dict[NUM_LIST_2] = epsilon_list_2
epsilon_dict[NUM_LIST_3] = epsilon_list_3
print("Epsilon_dict: " ,epsilon_dict)

overall_space = (MAX_STATE, MAX_ACTION)
q_table = np.zeros(overall_space)

#
print('\n')

file_path = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/SAMPLE_FROM_DICT/"
# file_name = "model.csv"
# file_name = "modified_model_dict2.csv"
file_name = "fixed_model_dict.csv"
# file_name = "fixed_model_dict_cifar10.csv"
file_csv = file_path + file_name
with open(file_csv) as f:
    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data  = [r for r in reader]
print('\n\n')
counter = 0

def check_equal(some_list):
    for i in range(len(some_list)):
        if some_list[i] != some_list[0]:
            return False
    return True

def match_epsilon(epsilon):
    # print("____________epsilon: ", epsilon)
    if epsilon+1 <= NUM_MODEL_1:
        # print("+++++++++++++++++++++++++")
        # print(epsilon_dict[NUM_LIST_1])
        return epsilon_dict[NUM_LIST_1]
    elif epsilon+1 > NUM_MODEL_1 and epsilon+1 <= NUM_MODEL_1+NUM_MODEL_2:
        return epsilon_dict[NUM_LIST_2][epsilon-NUM_MODEL_1]
    else:
        return epsilon_dict[NUM_LIST_3][epsilon-NUM_MODEL_1-NUM_MODEL_2]

def choose_action(num_layer, epsilon):
    # print("# EPILON: ",epsilon)
    eps = match_epsilon(epsilon)
    # print("eps inside choose_action fn: ", eps)
    if random.uniform(0,1) < eps:
        # get random keys of q_table[layer]
        random_key, random_value = random.choice(list(enumerate(q_table[num_layer])))
        return random_key,random_value
    else:
        if check_equal(q_table[num_layer]):
            max_key, max_value = random.choice(list(enumerate(q_table[num_layer])))
        else:
            max_key = q_table[num_layer].argmax()
            max_value = q_table[num_layer][max_key]
        return max_key,max_value

def choose_action_exp(num_layer,epsilon,index):
    eps = match_epsilon(epsilon)
    # print("eps : ",eps)
    # print(data[index])
    if epsilon < 1500:

        if num_layer ==0:
            index = random.randint(1,len(data)-1)

        action = str(data[index][num_layer+1])
        for x in Action:
            m = 'None'
            if action == x.name:
                m = x.value
                break
        if m == 'None':
            return
        value = q_table[num_layer][m]
        return m, value, index
    else:
        index = 0
        # print("epsilon: ", epsilon)
        action, value = choose_action(num_layer, epsilon)
        return action,value,index

def fn_format_action_array(action_array):
    new_action_array = action_array[:]
    length_action_array = len(action_array)
    if length_action_array != 4:
        for i in range(4-length_action_array):
            new_action_array.append('-')
    return new_action_array

def get_from_mem_replay(action_array):
    random_num = random.randint(1,len(data[1:])-1)
    print("random_num: ",random_num)
    print("accuracy: ",data[random_num][-2])
    return float(data[random_num][-2])
    # return 1
def update_qtable_from_mem_replay(num_model):

    for i in range(num_model):
        index = 0
        num_layer = 0
        action_array = []
        action_array_1 = []
        action = 0
        num_layer = 0
        alpha = alpha_list[i]
        # print("alpha= ",alpha)
        while Action(action).name != 's' and num_layer < 4:
            action,value_action, index = choose_action_exp(num_layer,i,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
            # print("action_array_1: ",action_array_1)
            max_value_next_action = get_next_value(num_layer, action_array_1)
            if num_layer == 3 and action_array_1[3] == 's':
                max_value_next_action = max_value_next_action * 0.2
            if num_layer == 2 and action_array_1[2] == 's':
                max_value_next_action = max_value_next_action * 0.2
            if num_layer == 1 and action_array_1[1] == 's':
                max_value_next_action = max_value_next_action * 0.2
            if num_layer == 0 and action_array_1[0] == 's':
                max_value_next_action = max_value_next_action * 0.2

            q_table[num_layer][action] = q_table[num_layer][action] + \
                        alpha*(gamma*max_value_next_action-q_table[num_layer][action])
            q_table[num_layer] = round_value(q_table[num_layer])
            num_layer += 1
    print(q_table)
def train_new_model(action_array):
    print("_________________ CANNOT FIND A MATCH _______________")
    print("______________________________________________________")
    # return 1
    # append_new_model(action_array)
    # print("+++++++++++++++++++ train(action_array) :",train(action_array))
    # return 0.97
    update_qtable_from_mem_replay(100)
    # print(q_table)
    # return random.uniform(0.6,0.99)
    # return 1
    # print("action_array: ",action_array)
    # return get_from_mem_replay(action_array)
    return train(action_array)

def get_accuracy(action_array):
    new_action_array = action_array[:]
    temp_action_array =  []
    temp_action_array = fn_format_action_array(new_action_array)
    for index in range(len(data)):

        if np.array_equal(temp_action_array, data[index][1:-2]):
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
            print("+++++++++ THERE IS A MATCH !! +++++++++++++++++")
            print("at model number: ", data[index][0])
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
            print("Accuray of model: ",data[index][-2])
            # print("''''''''''''''''''''''''''''''",q_table)
            global counter
            counter += 1
            return float(data[index][-2])
    return train_new_model(temp_action_array)

def get_next_value(num_layer,action_array):
    '''set num_layer to next layer '''
    num_layer += 1
    if num_layer == 4 or action_array[-1] == 's':
        return get_accuracy(action_array)
    else:
        max_key = q_table[num_layer].argmax()
        max_value = q_table[num_layer][max_key]
        return max_value

def translate_action_array(action_array):
    temp_array = []
    for i in range(len(action_array)):
        temp_array.append(Action(action_array[i]).name)
    return temp_array

def round_value(temp_q_table):
    return np.round(temp_q_table,6)

def save_q_table(episode,q_table):
    file_name = "q_table.csv"
    list_of_data = q_table[:]
    episode_name = 'episode_'+ str(episode)
    csv_columns = [[],[episode_name],['c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','m_1','m_2','m_3','s']]
    data_list = csv_columns + list(q_table)

    myFile = open(file_name,'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data_list)


for i in range(cont_episode,MAX_EPISODE):
    action = 0
    num_layer = 0
    alpha = alpha_list[i]
    action_array = []
    action_array_1 = []
    index = 0
    # print('alpha: ',alpha)
    while Action(action).name != 's' and num_layer < 4:
        # action,value_action = choose_action(num_layer, i)
        action,value_action, index = choose_action_exp(num_layer,i,index)
        action_array.append(action)
        # print("action_array: ",action_array)
        action_array_1 = translate_action_array(action_array)
        # print("action_array_1: ",action_array_1)
        # print("index: " ,index)
        max_value_next_action = get_next_value(num_layer, action_array_1)
        if num_layer == 3 and action_array_1[3] == 's':
            max_value_next_action = max_value_next_action * 0.2
        if num_layer == 2 and action_array_1[2] == 's':
            max_value_next_action = max_value_next_action * 0.2
        if num_layer == 1 and action_array_1[1] == 's':
            max_value_next_action = max_value_next_action * 0.2
        if num_layer == 0 and action_array_1[0] == 's':
            max_value_next_action = max_value_next_action * 0.2
        q_table[num_layer][action] = q_table[num_layer][action] + \
                    alpha*(gamma*max_value_next_action-q_table[num_layer][action])
        q_table[num_layer] = round_value(q_table[num_layer])
        # print(" num_layer : ",num_layer)
        num_layer += 1

    print(action_array_1)
    print("$$$$$$$$$$$$$$ EPISODE: ", i, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(q_table)
    save_q_table(i,q_table)
    print('\n')
    # time.sleep(1)

def get_best_action(table):
    dict_1 = {}
    tup_1 = ()
    for i in range(MAX_STATE):
        max_key = table[i].argmax()
        max_value = table[i][max_key]
        max_value = '%.5f' % round(max_value,5)
        tup_1 = (Action(max_key).name, max_value)
        key = "Layer "+ str(i+1)
        dict_1[key] = tup_1
    return dict_1

def get_avg_accuray(table):
    dict_1 = {}
    for i in range(MAX_STATE):
        key = "Layer "+ str(i+1)
        avg_value = sum(table[i])/len(table[i])
        avg_value = '%.5f' % round(avg_value, 5)
        dict_1[key] = avg_value
    return dict_1
print('----------------END------------------------------')
print('\n')
print(q_table)
print("NUMBER of model matched: ", counter)
print("Best accuracy: ",get_best_action(q_table))
print("Avg accuracy: ",get_avg_accuray(q_table))
print('\n')
print('\n')
