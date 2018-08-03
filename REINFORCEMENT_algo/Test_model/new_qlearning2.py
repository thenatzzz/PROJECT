import numpy as np
import random
import csv
import pandas
import os

from enum import Enum
from train_model2 import train

cont_episode = 1459

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
NUM_MODEL_3 = 500
MAX_EPISODE = NUM_MODEL_1 + NUM_MODEL_2 + NUM_MODEL_3

''' Discount Rate is set to 1.0 as to NOT prioritize any specific layer'''
gamma = 1.0

''' Alpha = learning rate '''
MIN_ALPHA = 0.1
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

#episode = 634
# cont_table = [[0.816386, 0.894554, 0.84094,  0.87199,  0.900976, 0.849807, 0.844385, 0.868581,
#   0.878616, 0.899242, 0.825518, 0.831804, 0.889203, 0.845373, 0.750034, 0.898981],
#  [0.901712, 0.849534, 0.920428, 0.892015, 0.911172, 0.878092, 0.866792, 0.858122,
#   0.852677, 0.847214, 0.883721, 0.916428, 0.903103, 0.893262, 0.852674, 0.939066],
#  [0.874732, 0.883285, 0.913511, 0.903967, 0.861925, 0.812135, 0.888467, 0.856509,
#   0.87668, 0.88427,  0.905238, 0.868618, 0.854658, 0.850311, 0.917834, 0.944988],
#  [0.893408, 0.950611, 0.932815, 0.902309, 0.921903, 0.891898, 0.889985, 0.888773,
#   0.91741,  0.915161, 0.924286, 0.855604, 0.934425, 0.913714, 0.850664, 0.901817]]
# q_table = np.asarray(cont_table)

#episode = 900
# cont_table = [[0.912253 0.937215 0.924066 0.936905 0.93166  0.913747 0.917221 0.918741
#   0.92961  0.935172 0.923351 0.920143 0.933461 0.925816 0.920636 0.908854]
#  [0.938487 0.929931 0.947038 0.936326 0.946355 0.941155 0.932449 0.933789
#   0.912853 0.936717 0.942204 0.943968 0.931551 0.927384 0.93022  0.948943]
#  [0.918071 0.936342 0.940082 0.93803  0.936982 0.932727 0.926288 0.935909
#   0.944096 0.924547 0.943275 0.937755 0.942384 0.919508 0.946118 0.95208 ]
#  [0.939785 0.906323 0.931328 0.91366  0.945843 0.91798  0.865624 0.9341
#   0.925519 0.950501 0.953994 0.923931 0.946823 0.953023 0.891244 0.946011]]

#episode = 750
# cont_table = [[0.848105 0.912665 0.907143 0.90727  0.915801 0.88859  0.889192 0.889863
#   0.909034 0.920528 0.895216 0.866935 0.909462 0.881674 0.875583 0.907678]
#  [0.932072 0.893226 0.933248 0.913253 0.932364 0.914978 0.906578 0.919287
#   0.877476 0.892165 0.915569 0.921637 0.914542 0.913776 0.895291 0.93657 ]
#  [0.90006  0.918612 0.929383 0.921239 0.906524 0.894981 0.91637  0.903417
#   0.917844 0.905981 0.930687 0.900389 0.907569 0.876258 0.933066 0.942397]
#  [0.923111 0.860659 0.939566 0.917029 0.938574 0.91033  0.830509 0.915205
#   0.927927 0.941703 0.950563 0.897266 0.941864 0.939872 0.858828 0.943761]]

#epsisode = 1097
# cont_table = [[0.936563, 0.946708, 0.939395, 0.946289, 0.937749, 0.942065, 0.938741, 0.932626,
#   0.940503, 0.946824, 0.946104, 0.939947, 0.944316, 0.942118, 0.942754, 0.910796],
#  [0.948103, 0.948258, 0.949608, 0.947775, 0.950617, 0.948833, 0.946043, 0.945354,
#   0.938889, 0.947984, 0.946981, 0.951203, 0.945438, 0.94879,  0.942886, 0.920396],
#  [0.944866, 0.954806, 0.952196, 0.951561, 0.949974, 0.952629, 0.944229, 0.951306,
#   0.949018, 0.93821,  0.955869, 0.951529, 0.951756, 0.947784, 0.954731, 0.931348],
#  [0.96341,  0.935449, 0.957812, 0.919628, 0.949189, 0.939815, 0.916516, 0.946848,
#   0.94119,  0.961428, 0.923764, 0.927118, 0.919142, 0.904985, 0.907863, 0.960699]]
# q_table = np.asarray(cont_table)


#episode = 1459
cont_table = [[0.955435, 0.957472, 0.956297, 0.954882, 0.956421, 0.954856, 0.955449, 0.953587,
  0.957928, 0.958485, 0.95746,  0.957602, 0.956114, 0.957306, 0.957552, 0.911055],
 [0.959879, 0.958457, 0.959507, 0.959317, 0.959555, 0.957393, 0.959396, 0.959786,
  0.959639, 0.958891, 0.959347, 0.959486, 0.959566, 0.957523, 0.958275, 0.9363  ],
 [0.959513, 0.959967, 0.96059,  0.958936, 0.959796, 0.960087, 0.959396, 0.959737,
  0.960738, 0.95901,  0.959941, 0.958069, 0.959577, 0.958732, 0.960097, 0.951585],
 [0.931555, 0.930566, 0.937688, 0.944561, 0.950189, 0.896932, 0.92814,  0.955556,
  0.913344, 0.961782, 0.961773, 0.956329, 0.953107, 0.924273, 0.938105, 0.961036]]
q_table = np.asarray(cont_table)
print('\n')

file_path = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/Test_model/"
# file_name = "model.csv"
file_name = "model_dict2.csv"
file_csv = file_path + file_name
with open(file_csv) as f:
    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data  = [r for r in reader]
# print("data: ",data)
print('\n\n')
counter = 0

def check_equal(some_list):
    for i in range(len(some_list)):
        if some_list[i] != some_list[0]:
            return False
    return True

def match_epsilon(epsilon):
    print("____________epsilon: ", epsilon)
    if epsilon+1 <= NUM_MODEL_1:
        # print("+++++++++++++++++++++++++")
        print(epsilon_dict[NUM_LIST_1])
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

def fn_format_action_array(action_array):
    new_action_array = action_array[:]
    length_action_array = len(action_array)
    if length_action_array != 4:
        for i in range(4-length_action_array):
            new_action_array.append('-')
    return new_action_array

def train_new_model(action_array):
    print("_________________ CANNOT FIND A MATCH _______________")
    # return 0
    # append_new_model(action_array)
    # print("+++++++++++++++++++ train(action_array) :",train(action_array))
    # return random.uniform(0.8,1)
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

for i in range(cont_episode,MAX_EPISODE):
    action = 0
    num_layer = 0
    alpha = alpha_list[i]
    action_array = []
    action_array_1 = []

    while Action(action).name != 's' and num_layer < 4:
        action,value_action = choose_action(num_layer, i)
        action_array.append(action)
        action_array_1 = translate_action_array(action_array)

        max_value_next_action = get_next_value(num_layer, action_array_1)

        q_table[num_layer][action] = q_table[num_layer][action] + \
                    alpha*(gamma*max_value_next_action-q_table[num_layer][action])
        q_table[num_layer] = round_value(q_table[num_layer])
        print(" num_layer : ",num_layer)
        num_layer += 1

    print(action_array_1)
    print("$$$$$$$$$$$$$$ EPISODE: ", i, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(q_table)
    print('\n')

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
