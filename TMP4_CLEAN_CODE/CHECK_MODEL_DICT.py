from HELPER_FUNCTION import get_data_from_csv, format_data_without_header
import statistics

ACCURACY_COL_INDEX = -2
LOSS_COL_INDEX = -1

def get_list(data,target):
    final_list = []
    for i in range(len(data)):
        if target == "accuracy":
            final_list.append(float(data[i][ACCURACY_COL_INDEX]))
        else:
            final_list.append(float(data[i][LOSS_COL_INDEX]))
    return final_list

def get_mean(data,target):
    final_list = get_list(data,target)
    mean = statistics.mean(final_list)
    return mean

def get_var(data,target):
    final_list = get_list(data,target)
    var = statistics.variance(final_list)
    return var

def get_standard_deviation(data,target):
    final_list  = get_list(data,target)
    std_dev = statistics.stdev(final_list)
    return std_dev

def get_accuracy_model(model):
    return model[ACCURACY_COL_INDEX]

def get_loss_model(model):
    return model[LOSS_COL_INDEX]

def format_data_without_header(data):
    return data[1:]

def get_best_topology(data,target):
    best_model = data[0]
    best_model_acc = get_accuracy_model(best_model)
    current_model = ""

    for i in range(len(data)):
        current_model = data[i]
        current_model_acc = get_accuracy_model(current_model)
        if current_model_acc > best_model_acc:
            best_model_acc = current_model_acc
            best_model = current_model
    print(best_model)
    return best_model

def get_worst_topology(data,target):
    worst_model = data[0]
    worst_model_acc = get_accuracy_model(worst_model)
    current_model = ""

    for i in range(len(data)):
        current_model = data[i]
        current_model_acc = get_accuracy_model(current_model)
        if current_model_acc < worst_model_acc:
            worst_model_acc = current_model_acc
            worst_model = current_model
    print(worst_model)
    return worst_model

def main():
    file_name = "fixed_model_dict.csv"
    file_name = "COMPLETE_CIFAR10.csv"
    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    target = "accuracy"
    target = "loss"

    best_model = get_best_topology(data,target)
    worst_model = get_worst_topology(data,target)

    print("Std: ",get_standard_deviation(data,target))
    print("Mean: ",get_mean(data,target))
    print("Variance: ",get_var(data,target))

if __name__ == "__main__":
    main()
