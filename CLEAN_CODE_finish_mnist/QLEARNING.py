from TRAIN_MODEL_MNIST import train

import numpy as np
import random
import csv
import pandas
import os
# from modified_train_model import train
from enum import Enum

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
NUM_MODEL_1 = 2000
NUM_MODEL_2 = 1000
NUM_MODEL_3 = 800

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
# cont_table = [[0.826234, 0.88426,  0.84127,  0.842995, 0.845733, 0.82823,  0.835235, 0.799535,
#   0.821951, 0.855575, 0.844262, 0.830372, 0.7287,   0.683016, 0.73654,  0.051434],
#  [0.878737, 0.885386, 0.894702, 0.892757, 0.900568, 0.920244, 0.898087, 0.894174,
#   0.891935, 0.898153, 0.890215, 0.878306, 0.88461,  0.875414, 0.887912, 0.102775],
#  [0.917463, 0.928613, 0.947393, 0.933518, 0.930839, 0.923504, 0.936717, 0.930549,
#   0.923629, 0.907756, 0.9362,   0.926688, 0.914589, 0.914352, 0.924581, 0.186038],
#  [0.958209, 0.937628, 0.952526, 0.947634, 0.949689, 0.947292, 0.935899, 0.951767,
#   0.941856, 0.95124,  0.94776,  0.947177, 0.935919, 0.941927, 0.95047,  0.190912]]
# q_table = np.asarray(cont_table)

print('\n')

def open_file(file_name):
    # file_path = "/homes/nj2217/PROJECT/CLEAN_CODE/"
    # file_name = "fixed_model_dict.csv"
    # file_csv = file_path + file_name
    # with open(file_csv) as f:
    with open(file_name) as f:
        reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                            quoting = csv.QUOTE_MINIMAL)
        data  = [r for r in reader]
    return data
    print('\n\n')


counter = 0

def check_equal(some_list):
    for i in range(len(some_list)):
        if some_list[i] != some_list[0]:
            return False
    return True

def match_epsilon(epsilon):
    if epsilon+1 <= NUM_MODEL_1:
        return epsilon_dict[NUM_LIST_1]
    elif epsilon+1 > NUM_MODEL_1 and epsilon+1 <= NUM_MODEL_1+NUM_MODEL_2:
        return epsilon_dict[NUM_LIST_2][epsilon-NUM_MODEL_1]
    else:
        return epsilon_dict[NUM_LIST_3][epsilon-NUM_MODEL_1-NUM_MODEL_2]

def choose_action(num_layer, epsilon):
    eps = match_epsilon(epsilon)
    if random.uniform(0,1) < eps:
        random_key, random_value = random.choice(list(enumerate(q_table[num_layer])))
        return random_key,random_value
    else:
        if check_equal(q_table[num_layer]):
            max_key, max_value = random.choice(list(enumerate(q_table[num_layer])))
        else:
            max_key = q_table[num_layer].argmax()
            max_value = q_table[num_layer][max_key]
        return max_key,max_value

def choose_action_exp(data,num_layer,epsilon,index):
    # print("index:------------- ", index)

    if epsilon < NUM_MODEL_1:
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
        # index += 1
        return m, value, index
    else:
        index = 0
        action, value = choose_action(num_layer, epsilon)
        return action,value,index


def fn_format_action_array(action_array):
    new_action_array = action_array[:]
    length_action_array = len(action_array)
    if length_action_array != 4:
        for i in range(4-length_action_array):
            new_action_array.append('-')
    return new_action_array

def get_from_mem_replay(data,action_array):
    random_num = random.randint(1,len(data[1:])-1)
    return float(data[random_num][-2])
    # return 1
def fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1):
    if num_layer == 3 and action_array_1[3] == 's':
        max_value_next_action = max_value_next_action * 0.2
    if num_layer == 2 and action_array_1[2] == 's':
        max_value_next_action = max_value_next_action * 0.2
    if num_layer == 1 and action_array_1[1] == 's':
        max_value_next_action = max_value_next_action * 0.2
    if num_layer == 0 and action_array_1[0] == 's':
        max_value_next_action = max_value_next_action * 0.2
    return max_value_next_action

def update_qtable_from_mem_replay(data,num_model):
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
            action,value_action, index = choose_action_exp(data,num_layer,i,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
            # print("action_array_1: ",action_array_1)
            max_value_next_action = get_next_value(data,num_layer, action_array_1)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            q_table[num_layer][action] = q_table[num_layer][action] + \
                        alpha*(gamma*max_value_next_action-q_table[num_layer][action])
            q_table[num_layer] = round_value(q_table[num_layer])
            num_layer += 1


def train_new_model(data,action_array):
    print("_________________ CANNOT FIND A MATCH _______________")
    print("______________________________________________________")
    # return 1
    # append_new_model(action_array)
    # print("+++++++++++++++++++ train(action_array) :",train(action_array))
    # return 0.97
    # return 0
    # num_model = 50

    num_model = 10
    # update_qtable_from_mem_replay(data,num_model)

    # return 1
    # return random.uniform(0.92,0.99)
    # print("action_array: ",action_array)
    # return get_from_mem_replay(data,action_array)
    return train(data,action_array)

def get_accuracy(data,action_array):
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
            global counter
            counter += 1
            return float(data[index][-2])
    return train_new_model(data,temp_action_array)

def get_next_value(data,num_layer,action_array):
    '''set num_layer to next layer '''
    num_layer += 1
    if num_layer == 4 or action_array[-1] == 's':
        return get_accuracy(data,action_array)
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

def run_q_learning(data):

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
            action,value_action, index = choose_action_exp(data,num_layer,i,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
            print("action_array_1: ",action_array_1)

            max_value_next_action = get_next_value(data,num_layer, action_array_1)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            q_table[num_layer][action] = q_table[num_layer][action] + \
                        alpha*(gamma*max_value_next_action-q_table[num_layer][action])
            q_table[num_layer] = round_value(q_table[num_layer])
            num_layer += 1

        print(action_array_1)
        print("$$$$$$$$$$$$$$ EPISODE: ", i, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        save_q_table(i,q_table)

        print(q_table)
        print('\n')

    print("NUMBER of model matched: ", counter)
    print("Best accuracy: ",get_best_action(q_table))
    print("Avg accuracy: ",get_avg_accuray(q_table))
    return get_best_action(q_table)
