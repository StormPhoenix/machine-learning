def load_data_set_float(filename):
    data_array = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        split_line = line.strip().split('\t')
        float_array = list(map(float, split_line))
        data_array.append(float_array)
    return data_array


def load_data_set_str(filename):
    data_array = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        split_line = line.strip().split('\t')
        str_array = list(split_line)
        data_array.append(str_array)
    return data_array
