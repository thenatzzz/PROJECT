from QLEARNING import run_q_learning, open_file
from RANDOM_TOPOLOGY import get_random_topology
from TRAIN_MODEL_CIFAR10 import pre_train_model_cifar10
from tmpTRAIN_MODEL_MNIST import pre_train_model

if __name__ == "__main__":

    '''
    #Get random topologies then save to csv file
    random_topology_file  = 'test_random_topology.csv'
    num_model = 1500
    file_name = get_random_topology(num_model, random_topology_file)
    print(file_name)
    pre_train_model(file_name)
    '''

    '''
    #Run Q-learning to find best topology
    file_name = "fixed_model_dict.csv"
    # file_name = "bad_model.csv"
    # file_name = "biased_dict.csv"
    # file_name = "COMPLETE_CIFAR10.csv"
    data = open_file(file_name)
    best_topology = run_q_learning(data)
    print("best_topology: ", best_topology)
    # accuracy, loss = to_verify_model(best_topology)
    '''


    #Get random topologies then save to csv file
    # random_topology_file  = 'test_random_topology.csv'
    # num_model = 1500
    # file_name = get_random_topology(num_model, random_topology_file)
    # print(file_name)
    # pre_train_model_cifar10(file_name)

    #Run Q-learning to find best topology
    # file_name = "fixed_model_dict.csv"
    # file_name = "bad_model.csv"
    # file_name = "biased_dict.csv"
    file_name = "COMPLETE_CIFAR10.csv"
    data = open_file(file_name)
    best_topology = run_q_learning(data)
    print("best_topology: ", best_topology)
    # accuracy, loss = to_verify_model(best_topology)