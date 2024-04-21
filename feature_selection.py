import numpy as np
from pyds import MassFunction
from tabulate import tabulate

def calculate_average_mass(mass):
    mass_mean = mass.mean(axis=0) 
    mass_median = np.median(mass, axis=0)

    return mass_mean, mass_median

def attach_feature_names(names, value):
    linked_theta_names = {}

    for k, v in zip(names, value):
        linked_theta_names[k] = v

    return linked_theta_names

def sort_dict(dict, reverse=False):
    sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=reverse)

    return sorted_dict

def output_feature_ranking(dict, method, type):
    feature_name_list = []
    feature_ranking_values = []

    for k, v in dict:
        feature_name_list.append(k)
        feature_ranking_values.append(v)

    headers = ["Feature", method]
    table = zip(feature_name_list, feature_ranking_values)

    features_file = open("feature_selection_" + method.lower() + "_ranking_" + type + ".txt", "a")

    features_file.writelines(tabulate(table, headers=headers, floatfmt=".4f"))
    features_file.close()

def measure_feature_correctness(theta, y_test):
    length = y_test.shape[0]

    theta = theta > 0.5
    theta = theta.astype(int)

    res = theta == y_test[:, None]
    res = res.astype(int)
    
    one_occurrences = np.count_nonzero(res == 1, axis = 0)
    one_occurrences = one_occurrences / length

    return one_occurrences