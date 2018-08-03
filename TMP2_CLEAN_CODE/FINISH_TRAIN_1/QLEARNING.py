# from TRAIN_MODEL_MNIST import train
from tmpTRAIN_MODEL_MNIST import train
from HELPER_FUNCTION import get_data_from_csv, format_data_without_header
from TRAIN_MODEL_CIFAR10 import train_model_cifar10
import numpy as np
import random
import csv
import pandas
import os
import time
# from modified_train_model import train
from enum import Enum

FILE_CIFAR10 = "COMPLETE_CIFAR10.csv"
FILE_MNIST = "fixed_model_dict.csv"
# MODE = "RANDOMIZED_update"
MODE = "SEQUENTIAL_update"
Q_TABLE_FILE= ""
# Q_TABLE_FILE= "q_table_cifar10_rd.csv"
Q_TABLE_FILE= "q_table_cifar10_sq_no_exp2248.csv"

# Q_TABLE_FILE= "q_table_mnist.csv"


cont_episode = 2248
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
#c_5,c_7,c_3,c_3
''' Create MODEL corresponding to specific epsilon '''
NUM_LIST = 3
NUM_LIST_1 = 1
NUM_LIST_2 = 2
NUM_LIST_3 = 3
''' Change this to specify number of model corresponding to each specific epsilon'''
NUM_MODEL_1 = 1800
NUM_MODEL_2 = 500
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
q_table = [[0.156619,	0.075328,	0.07581,	0.073396,	0.067358,	0.063142,	0.060533,	0.061195,	0.075718,	0.071361,	0.05312,	0.063628,	0.047864,	0.040744,	0.038188,	0.020336
],[0.144351,	0.139018,	0.144309,	0.136505,	0.150599,	0.15607,	0.282017,	0.160827,	0.151589,	0.146944,	0.136577,	0.142708,	0.139312,	0.136053,	0.145388,	0.049115
],[0.293787,	0.283745,	0.339795,	0.333108,	0.326737,	0.304493,	0.301681,	0.327434,	0.283981,	0.308326,	0.456368,	0.266033,	0.277322,	0.309883,	0.261587,	0.101534
],[0.519596,0.48249,	0.511011,	0.496835,	0.484195,	0.511916,	0.49581,	0.498631,	0.646515,	0.48636,0.518391,	0.515634,	0.490491,	0.529176,	0.534052,	0.115155]]

q_table = np.asarray(q_table)
print('\n')

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

# def initialize_q_table():
def choose_action_exp(data,num_layer,episode,tracking_index):
    # print("index:------------- ", tracking_index)
    # print("episode:------------- ", episode)
    # print("MODE inside choose_action_exp: ", MODE)
    if episode < NUM_MODEL_1:
        if num_layer ==0:
            if MODE == "RANDOMIZED_update":
                tracking_index = random.randint(0,len(data)-1)
            elif MODE == "SEQUENTIAL_update":
                tracking_index = episode
            else:
                tracking_index = random.randint(0,len(data)-1)
        action = str(data[tracking_index][num_layer+1])
        for enum_action in Action:
            tracking_action = 'None'
            if action == enum_action.name:
                tracking_action = enum_action.value
                break
        if tracking_action == 'None':
            return
        value = q_table[num_layer][tracking_action]
        # index += 1
        return tracking_action, value, tracking_index
    else:
        tracking_index = 0
        action, value = choose_action(num_layer, episode)
        return action,value,tracking_index

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

def update_qtable_from_mem_replay(data,num_model,dataset):
    global MODE
    MODE = "RANDOMIZED_update"
    for i in range(num_model):
        index = 0
        num_layer = 0
        action_array = []
        action_array_1 = []
        action = 0
        num_layer = 0
        alpha = alpha_list[i]
        # print("alpha= ",alpha)
        # print("MODE inside update qtable: ", MODE)
        while Action(action).name != 's' and num_layer < 4:
            action,value_action, index = choose_action_exp(data,num_layer,i,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
            # print("action_array_1: ",action_array_1)
            max_value_next_action = get_next_value(data,num_layer, action_array_1,dataset)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            q_table[num_layer][action] = q_table[num_layer][action] + \
                        alpha*(gamma*max_value_next_action-q_table[num_layer][action])
            q_table[num_layer] = round_value(q_table[num_layer])
            num_layer += 1


def train_new_model(data,action_array,dataset):
    print("_________________ CANNOT FIND A MATCH _______________")
    print("______________________________________________________")

    num_model = 5
    # update_qtable_from_mem_replay(data,num_model,dataset)

    # print("LENGTH_DATA :::::::::::::::::::: ", len(data))
    # return 1
    # return random.uniform(0.70,0.75)
    # print("action_array: ",action_array)
    # return get_from_mem_replay(data,action_array)
    if dataset == "cifar10":
        return train_model_cifar10(action_array)
    elif dataset == "mnist":
        return train(action_array)

def get_accuracy(data,action_array,dataset):
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
    return train_new_model(data,temp_action_array,dataset)

def get_next_value(data,num_layer,action_array,dataset):
    '''set num_layer to next layer '''
    num_layer += 1
    if num_layer == 4 or action_array[-1] == 's':
        return get_accuracy(data,action_array,dataset)
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

def save_q_table(episode,q_table,dataset):
    if len(Q_TABLE_FILE) > 0 :
        file_name = Q_TABLE_FILE
    elif dataset == "cifar10":
        file_name = "q_table_cifar10.csv"
    elif dataset == "mnist":
        file_name = "q_table_mnist.csv"

    list_of_data = q_table[:]
    episode_name = 'episode_'+ str(episode)
    csv_columns = [[],[episode_name],['c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','m_1','m_2','m_3','s']]
    data_list = csv_columns + list(q_table)

    myFile = open(file_name,'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data_list)

def update_data(data,dataset):
    tmp_data = ""
    if dataset == "cifar10":
        tmp_data = get_data_from_csv(FILE_CIFAR10)
        tmp_data = format_data_without_header(tmp_data)
        data = tmp_data[:]
        return data
    elif dataset == "mnist":
        tmp_data = get_data_from_csv(FILE_MNIST)
        tmp_data = format_data_without_header(tmp_data)
        data = tmp_data[:]
        return data

def run_q_learning(data,dataset):

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
            # print(i)
            max_value_next_action = get_next_value(data,num_layer, action_array_1,dataset)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            q_table[num_layer][action] = q_table[num_layer][action] + \
                        alpha*(gamma*max_value_next_action-q_table[num_layer][action])
            q_table[num_layer] = round_value(q_table[num_layer])
            num_layer += 1
        # print("ZZZZZZZ index ZZZZZZZZZZZZZ: ", index)
        print(action_array_1)
        print("$$$$$$$$$$$$$$ EPISODE: ", i, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        save_q_table(i,q_table,dataset)
        data = update_data(data,dataset)
        print(q_table)
        print('\n')
        # time.sleep(5)
    print("NUMBER of model matched: ", counter)
    print("Best accuracy: ",get_best_action(q_table))
    print("Avg accuracy: ",get_avg_accuray(q_table))
    return get_best_action(q_table)
