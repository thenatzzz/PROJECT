from QLEARNING import run_q_learning, open_file
# from VERIFY_MODEL_MNIST import train
# from TRAIN_MODEL_MNIST import train
from RANDOM_TOPOLOGY import get_random_topology
# from TRAIN_MODEL_MNIST import pre_train_model, to_verify_model


if __name__ == "__main__":
    # random_topology_file  = 'test_random_topology.csv'
    # num_model = 1500
    # file_name = get_random_topology(num_model, random_topology_file)
    # print(file_name)
    # pre_train_model(file_name)

    file_name = "fixed_model_dict.csv"
    # file_name = "bad_model.csv"
    # file_name = "biased_dict.csv"
    data = open_file(file_name)
    best_topology = run_q_learning(data)
    print("best_topology: ", best_topology)
    # accuracy, loss = to_verify_model(best_topology)

    #
    # first_layer = best_topology['Layer 1']
    # print("first_layer:",first_layer)
    # second_layer = best_topology['Layer 2']
    # print("second_layer:",second_layer)
    #
    # third_layer = best_topology['Layer 3']
    # print("third_layer:",third_layer)
    #
    # if third_layer[0] == 's':
    #     forth_layer = '-'
    # else:
    #     forth_layer = best_topology['Layer 4']
    # print("forth_layer:",forth_layer)
    #
    #
    # train([first_layer[0], second_layer[0], third_layer[0], forth_layer[0]])
