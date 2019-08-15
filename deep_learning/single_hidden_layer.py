import os
import pickle

import numpy as np

HIDDEN_LAYER_COUNT = 2
HIDDEN_LAYER_UNITS = [5, 4]
# HIDDEN_LAYER_UNITS = [5, 10, 20, 40, 60, 30, 20, 5]

RANDOM_MODEL_PATH = '../deep_learning/random.model'
FINAL_MODEL_PATH = '../deep_learning/final.model'

iteration_count = 100

alpha = 0.1

TRAIN_DATA_PATH = '../deep_learning/testSetRBF.txt'
TEST_DATA_PATH = '../deep_learning/testSetRBF2.txt'


def store_model(model, file_path):
    fw = open(file_path, "wb")
    pickle.dump(model, fw)
    fw.close()


def load_model(file_path):
    fr = open(file_path, "rb")
    return pickle.load(fr)


def load_data(file_path=TRAIN_DATA_PATH):
    data_set = []
    label = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            line_array = line.strip().split('\t')
            data_set.append([float(line_array[0]), float(line_array[1])])
            label.append(float(line_array[2]))
    data_set = np.mat(data_set)
    label = np.mat(label).T
    row_index, _ = np.nonzero(label < 0)
    label[row_index] = 0
    return data_set, label


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_network_framework(feature_count, hidden_layer_count=HIDDEN_LAYER_COUNT, unit_count_list=HIDDEN_LAYER_UNITS):
    if len(unit_count_list) != hidden_layer_count:
        return None
    weight_list = []
    bias_list = []
    input_dimension = feature_count
    for i in range(hidden_layer_count):
        unit_count = unit_count_list[i]
        weight = np.mat(np.random.rand(unit_count, input_dimension))
        b = np.random.rand(unit_count, 1)
        weight_list.append(weight)
        bias_list.append(b)
        input_dimension = unit_count
    # output layer
    weight = np.mat(np.random.rand(1, input_dimension))
    b = np.mat(np.random.rand(1, 1))
    weight_list.append(weight)
    bias_list.append(b)
    return {'weight': weight_list, 'bias': bias_list, 'layers': hidden_layer_count + 1}


# TODO split data into batch
def make_single_hidden_layer_network(data, label):
    x = np.mat(data).T
    y = label.T
    feature_count, data_count = x.shape
    network = make_network_framework(feature_count=feature_count)
    weight_list = network['weight']
    bias_list = network['bias']
    store_model({'weight': weight_list, 'bias': bias_list}, RANDOM_MODEL_PATH)
    layers = network['layers']
    print(weight_list)
    z_cache = [x]
    for layer in range(HIDDEN_LAYER_COUNT):
        z = np.mat(np.zeros((HIDDEN_LAYER_UNITS[layer], data_count)))
        z_cache.append(z)
    # output z cache
    z_cache.append(np.mat(np.zeros((1, data_count))))

    for iter_count in range(iteration_count):
        # print('iteration count: ', iter_count)
        a = z_cache[0]
        for layer in range(layers):
            weight = weight_list[layer]
            bias = bias_list[layer]
            z = weight * a + bias
            a = sigmoid(z)
            z_cache[layer + 1] = z
        # output layer a
        dai = - (y / a) + (1 - y) / (1 - a)
        for layer in range(layers - 1, -1, -1):
            ai = sigmoid(z_cache[layer + 1])
            ai_1 = sigmoid(z_cache[layer])
            weight = weight_list[layer]
            bias = bias_list[layer]
            # calulate dw
            dw = np.mat(np.zeros(weight.shape))
            db = np.mat(np.zeros(bias.shape))
            base = np.multiply(dai, np.multiply(ai, (1 - ai)))
            for i in range(data_count):
                if i == 99:
                    print('')
                dw = dw + np.dot(base[:, i], ai_1[:, i].T)
                db = db + base[:, i]
            dw = dw / data_count
            print('iteration: ', iter_count)
            print('dw: ', dw)
            db = db / data_count
            # update da
            dai = np.dot(weight.T, base)
            # update weight
            weight_list[layer] = weight - alpha * dw
            # update db
            bias_list[layer] = bias - alpha * db
    return weight_list, bias_list


def predict(weight_list, bias_list, data):
    if len(weight_list) != len(bias_list):
        print('Predict Error. ')
        return None
    layers = len(weight_list)
    value = np.mat(data).T
    for layer in range(layers):
        weight = weight_list[layer]
        bias = bias_list[layer]
        value = sigmoid(np.dot(weight, value) + bias)
    return value


def main():
    if not os.path.exists(FINAL_MODEL_PATH):
        train_set, train_label = load_data(TRAIN_DATA_PATH)
        weight_list, bias_list = make_single_hidden_layer_network(train_set, train_label)
        store_model({'weight': weight_list, 'bias': bias_list}, FINAL_MODEL_PATH)
        print(weight_list)

    test_set, test_label = load_data(TEST_DATA_PATH)
    test_count, _ = test_set.shape
    correct_count = 0
    network = load_model(RANDOM_MODEL_PATH)
    # network = load_model(FINAL_MODEL_PATH)
    weight_list = network['weight']
    bias_list = network['bias']
    for i in range(test_count):
        value = predict(weight_list, bias_list, test_set[i, :])
        real_value = test_label[i, 0]
        if value > 0.5:
            value = 1
        else:
            value = 0
        if value == real_value:
            correct_count += 1
    print(correct_count, test_count, float(correct_count) / test_count)


if __name__ == '__main__':
    main()
