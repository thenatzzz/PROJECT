import numpy as np
import random
import csv
random.seed(42)

MAX_LAYER = 4

NUM_EPISODE = 10
MIN_ALPHA = 0.1
eps = 0.2
gamma = 1.0
alphas = np.linspace(1.0, MIN_ALPHA, NUM_EPISODE)
print("alphas_table: ")
print(alphas)
print('\n')
''' Initialize q tables '''
q_table = np.zeros([4,16])
q_table = np.random.rand(4,16)
print(q_table)
# print(q_table)
# print(q_table[0])

file_path = "/homes/nj2217/PROJECT/tensorflow_tut/"
file_name = "MODEL_dict.csv"
file_csv = file_path + file_name
with open(file_csv) as f:
    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data  = [r for r in reader]
    #data = data[1:]      # To get rid of column name
print(file_csv)
#print(data)
print('\n\n')

def get_num_layer(episode):
    return 4
# def get_accuracy():

def act(layer,action):
    done = True
    next_layer = layer + 1
    # reward = 0

    if layer == 4:
        done = False
        next_layer = layer
        # reward = get_accuracy()
    # return next_layer, reward,done
    return next_layer, done

def choose_action(layer):
    if random.uniform(0,1) < eps:
    #    print(q_table[layer])
    #    print(random.choice(q_table[layer]))

        return random.choice(q_table[layer])
    else:
        return np.argmax(q_table[layer])

for i in range(NUM_EPISODE):
    num_layer = 1
    layer  = 0
    total_reward = 0
    alpha = alphas[i]
    action_array = []
    while num_layer != 4 or layer != 's':
    # for layer in range(num_layer):
        action = choose_action(layer)
        action_array.append(action)
#        print("action_array: ",action_array)
        next_layer, done = act(layer,action)
        # reward = get_accuracy(action_array)
        reward = 0
        total_reward += reward
        action = int(action)
        q_table[layer][action] = q_table[layer][action] + \
                            alpha*(gamma * np.max(q_table[next_layer])-q_table[layer][action])
        layer = next_layer
#        print('\n')

        if done:
#            print('IN if_done ->layer: ', layer)
            break
        num_layer += 1
    print(f"Episode {i + 1}: total reward -> {total_reward}")
