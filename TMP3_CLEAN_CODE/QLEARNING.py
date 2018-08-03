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
#PBS -l select=1:ncpus=8:mpiprocs=4

# from modified_train_model import train
from enum import Enum

FILE_CIFAR10 = "COMPLETE_CIFAR10.csv"
FILE_MNIST = "fixed_model_dict.csv"
# MODE = "RANDOMIZED_update"
MODE = "SEQUENTIAL_update"
Q_TABLE_FILE= ""
# Q_TABLE_FILE= "q_table_cifar10_rd.csv"
Q_TABLE_FILE= "q_table_cifar10_sq_with_exp_2637.csv"

# Q_TABLE_FILE= "q_table_mnist.csv"


cont_episode = 2637
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
NUM_MODEL_1 = 2000
NUM_MODEL_2 = 800
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
# print("Epsilon_dict: " ,epsilon_dict)

overall_space = (MAX_STATE, MAX_ACTION)
q_table = np.zeros(overall_space)
q_table = [[0.483470,0.553176,0.462418,0.429110,0.490972,0.481987,0.464253,0.412064,0.453932,0.473117,0.464815,0.447106,0.232867,0.310908,0.281464,0.039308],
[0.513834,0.539195,0.540991,0.514320,0.532735,0.632903,0.541123,0.534079,0.528802,0.545651,0.536041,0.518831,0.538766,0.517206,0.530916,0.08028],
[0.617062,0.603490,0.628797,0.619212,0.611157,0.619007,0.611953,0.592881,0.596563,0.626524,0.681857,0.598665,0.603247,0.610896,0.614264,0.13551],
[0.699671,0.680950,0.691868,0.656772,0.681709,0.713378,0.692969,0.683870,0.701570,0.689740,0.691647,0.696842,0.676952,0.690272,0.680156,0.138568]]
q_table =np.asarray(q_table)
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
    update_qtable_from_mem_replay(data,num_model,dataset)

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
