import csv
import os

def open_file(file_csv):
    with open(file_csv) as f:
        reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
        data  = [r for r in reader]
        data = data[1:]      # To get rid of column name
    return data

def save_file(data,csv_file):
    csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy', 'Loss']
    path_to_file = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/SAMPLE_FROM_DICT/"
    csv_file = path_to_file+"fixed_model_dict_cifar10.csv"
    list_of_dict = []
    # print(data)
    # print("TYPE data:", type(data))
    for index in range(len(data)):
        temp_dict = {}
        temp_dict['Model'] = data[index][0]
        temp_dict['1st Layer'] = data[index][1]
        temp_dict['2nd Layer'] = data[index][2]
        temp_dict['3rd Layer'] = data[index][3]
        temp_dict['4th Layer'] = data[index][4]
        temp_dict['Accuracy'] = data[index][5]
        temp_dict['Loss'] = data[index][6]
        list_of_dict.append(temp_dict)

    try:
       with open(csv_file, 'a') as csvfile:

           writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
           if os.stat(csv_file).st_size == 0 : # ONLY write ROW Hedaer when file is empty
              writer.writeheader()
           for data in list_of_dict:
               writer.writerow(data)
    except IOError:
       print("I/O error")

def fix_dict(data):
    final_array = []

    for array in data:
        # final_array = []
        temp_array = array[:]
        array = array[1:-2]
        # print(array)
        count = 0
        for single_element in array:
            if single_element != '-':
                count += 1
            # print(single_element)
        if count < 4 and array[count-1] != 's':
            print("model: ", temp_array[0])
            print("count: ",count)
            print("found")
            print("Old array : ",array)
            array[count] = 's'
            print("Fixed array: ",array)
            temp_array[count+1] = 's'
            print("temp_array: ", temp_array)
            print('\n')
        final_array.append(temp_array)
    # print(final_array)
    return final_array
    # print(data[133])
        # print(array)

if __name__ == '__main__':
    file_name = "model_dict_cifar10.csv"
    new_file_name = "fixed_model_dict_cifar10.csv"

    data = open_file(file_name)
    data = fix_dict(data)
    # print(data)
    save_file(data, new_file_name)
    # print(data)
