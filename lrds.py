import numpy as np
import pandas as pd
from pyds import MassFunction
from tabulate import tabulate
from process import clean_dataset
import matplotlib.pyplot as plt

def calculate_weights(b, data, W, alpha):
    phi_features = data
    beta_features = W

    weights = (phi_features * beta_features) + alpha

    return weights


def calculate_positive_negative_parts(weights):
    positive_part = []
    negative_part = []

    # weights = weights.tolist()

    positive_part = np.maximum(0, weights)
    negative_part = np.maximum(0, -weights)

    return positive_part, negative_part


def sum_weights(positive_part, negative_part):
    '''
    Calculates the sum of all of the "positive part", and the "negative part" weights from p.9, Section 3.1.1 of [1].\n
    Returns two one-dimensional arrays, one "positive part" and one "negative part".
    '''
    positive_list = positive_part.sum(axis=1)
    negative_list = negative_part.sum(axis=1)

    return positive_list, negative_list


def calculate_mass_functions(positive_part, negative_part):
    ''' Returns 2 dimesntional matrices for θ1, θ2, Θ, κ. For example, for θ1 we have:
            f1       | f2        ...
   instance
        1   θ1_value | θ1_value  ...
        2   θ1_value | θ1_value  ...
        3   θ1_value | θ1_value  ...
    '''
    k = (1 - np.exp(-positive_part)) * (1 - np.exp(-negative_part))

    # if not reducing k, then use this:
    k_find_largest = np.where(k == 1, -1, k)
    k_find_largest = np.amax(k_find_largest)

    k = np.where(k == 1, k_find_largest, k)

    np.seterr(divide='ignore', invalid='ignore')
    theta1 = ((1 - np.exp(-positive_part)) * (np.exp(-negative_part))) / (1 - k)
    theta2 = ((1 - np.exp(-negative_part)) * (np.exp(-positive_part))) / (1 - k)
    theta = ((np.exp(-positive_part)) * (np.exp(-negative_part))) / (1 - k)

    return theta1, theta2, theta, k


def join_mass_functions(theta1, theta2, theta):
    mass_functions = []

    for t1, t2, tu in zip(theta1, theta2, theta):
        mass_functions.append(MassFunction({'1': t1, '2': t2, '12': tu}))

    mass_functions = np.asarray(mass_functions)

    return mass_functions


def calculate_plausibility(masses, theta_value):
    '''
    masses <array>: The list of masses calculated for each instance of the dataset.\n
    theta_value <string>: The flag, showing which value you eant to calculate plausibility for ('1' for  θ1 etc).\n

    Calculates the plausibility based on the theta value you select as a flag.
    '''

    plausibility_list = []
    masses = masses.flatten().tolist()

    for i in masses:
        plausibility_list.append(i.pl(theta_value))

    plausibility_array = np.asarray(plausibility_list)

    return plausibility_array


def calculate_belief(masses, theta_value):
    '''
    masses <array>: The list of masses calculated for each instance of the dataset.\n
    theta_value <string>: The flag, showing which value you eant to calculate plausibility for ('1' for  θ1 etc).\n

    Calculates the belief based on the theta value you select as a flag.
    '''

    belief_list = []
    masses = masses.flatten().tolist()

    for i in masses:
        belief_list.append(i.bel(theta_value))

    belief_array = np.asarray(belief_list)

    return belief_array


def decision_rule(theta1, theta2, rule):
    """
    Creates an output based on the rules from Thierry's paper §2.1.7

    theta1 <array>: numpy array of the Theta 1 values\n
    theta2 <array>: numpy array of the Theta 2 values
    """

    decision = []

    if rule != "maximal":
        for (theta1, theta2) in zip(theta1, theta2):
            if (1 - theta1) <= (1 - theta2):
                decision.append(1)
            else:
                decision.append(0)
    else:
        for (theta1, theta2) in zip(theta1, theta2):
            if (1 - theta1) < (1 - theta2):
                decision.append(1)
            else:
                decision.append(0)

    decision = np.asarray(decision)

    return decision


def get_interval(conservative_approach, maximal_element):
    """
    Creates an output based on the interval from the rules from Thierry's paper §2.1.7

    conservative_approach <array>: numpy array of the conservative approach decision rule\n
    maximal_element <array>: numpy array of the maximal element decision rule
    """

    decision = []

    for (c, m) in zip(conservative_approach, maximal_element):
        if c == 1 and m == 1:
            decision.append(1)
        elif c == 0 and m == 0:
            decision.append(0)
        else:
            decision.append(2)

    decision = np.asarray(decision)

    return decision

def remove_contradictory_instances(thetas_stack):
    thetas_df = pd.DataFrame(thetas_stack)
    thetas_df = clean_dataset(thetas_df)

    theta1_new = thetas_df.values[:, 0]
    theta2_new = thetas_df.values[:, 1]
    theta_new = thetas_df.values[:, 2]
    k_new = thetas_df.values[:, 3]
    y_test_new = thetas_df.values[:, -1]

    return theta1_new, theta2_new, theta_new, k_new, y_test_new

def normalise_data(data, rX):
    feature_mean = rX.mean()
    feature_std = rX.std()

    normalised_data = (data - feature_mean) / feature_std

    return normalised_data

def synthesise_data(feature1, feature2):
    synth_size = 750

    try:
        synth_feature1 = np.arange(np.amin(feature1), np.amax(feature1), (np.amax(feature1) - np.amin(feature1)) / synth_size)
        synth_feature2 = np.arange(np.amin(feature2), np.amax(feature2), (np.amax(feature2) - np.amin(feature2)) / synth_size)

        return synth_feature1, synth_feature2, synth_size
    except:
        pass

def create_figure_7(feature_value, theta1, theta2, theta, feature_name):
    print("Ordering values and mass functions for feature " + feature_name + "... ", end="", flush=True)
    raw_and_theta_values = np.column_stack((theta1, theta2, theta, feature_value))

    sorted_values = raw_and_theta_values[raw_and_theta_values[:, -1].argsort()]

    theta1 = sorted_values[:, 0]
    theta2 = sorted_values[:, 1]
    theta = sorted_values[:, 2]
    raw_values = sorted_values[:, -1]
    print(" Done!")

    print("Creating figure 7 for feature " + feature_name + "... ", end="", flush=True)
    plt.clf()
    plt.plot(raw_values, theta1, 'k-', label="{θ1}")
    plt.plot(raw_values, theta2, 'k--', label="{θ2}")
    plt.plot(raw_values, theta, 'k:', label="{θ1, θ2}")
    plt.xlabel(feature_name)
    plt.ylabel("Mass")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('img/' + feature_name.replace('/', '') + '.png')
    print(" Done!")

def create_figure_8 (W, b, D, rX, feature_names, y_test):
    feature1 = rX[:, 0]
    feature2 = rX[:, 1]

    synth_feature1, synth_feature2, synth_size = synthesise_data(feature1, feature2)

    X, Y = np.meshgrid(synth_feature1, synth_feature2)

    feature1_normalised = normalise_data(X, feature1)
    feature2_normalised = normalise_data(Y, feature2)

    alpha = b/D

    feature1_weight = calculate_weights(b, feature1_normalised, W[0], alpha)
    feature2_weight = calculate_weights(b, feature2_normalised, W[1], alpha)


    feature1_pos_part, feature1_neg_part = calculate_positive_negative_parts(feature1_weight)
    feature2_pos_part, feature2_neg_part = calculate_positive_negative_parts(feature2_weight)


    positive_part_summed = np.add(feature1_pos_part, feature2_pos_part)
    negative_part_summed = np.add(feature1_neg_part, feature2_neg_part)

    Z_theta1, Z_theta2, Z_theta, Z_k = calculate_mass_functions(positive_part_summed, negative_part_summed)

    joined_mass_functions = joined_mass_functions = join_mass_functions(Z_theta1.flatten(), Z_theta2.flatten(), Z_theta.flatten())

    Z_theta1_bel_synth = calculate_belief(joined_mass_functions, "1")
    Z_theta2_bel_synth = calculate_belief(joined_mass_functions, "2")

    Z_theta1_pl_synth = calculate_plausibility(joined_mass_functions, "1")
    Z_theta2_pl_synth = calculate_plausibility(joined_mass_functions, "2")

    conservative_approach_synth = decision_rule(Z_theta1_bel_synth, Z_theta2_pl_synth, "conservative")
    maximal_element_synth = decision_rule(Z_theta1_pl_synth, Z_theta2_bel_synth, "maximal")
    optimist_rule_synth = decision_rule(Z_theta1_pl_synth, Z_theta2_pl_synth, "optimist")

    conservative_approach_synth = conservative_approach_synth.reshape(synth_size, synth_size)
    maximal_element_synth = maximal_element_synth.reshape(synth_size, synth_size)
    optimist_rule_synth = optimist_rule_synth.reshape(synth_size, synth_size)
    


    # figure 8d
    plt.ylabel(feature_names[0])
    plt.xlabel(feature_names[1])
    
    plt.scatter(feature2[y_test==0], feature1[y_test==0], marker='o', s=30, facecolors='none', edgecolors='green')
    plt.scatter(feature2[y_test==1], feature1[y_test==1], marker='^', s=30, facecolors='none', edgecolors='red')

    # figure 8c
    Z = Z_k
    CS = plt.contour(Y, X, Z, levels=5, linestyles = 'dashed', colors = 'black')
    plt.clabel(CS, CS.levels, inline=True, fontsize=8)

    Z =  Z_theta
    XS = plt.contour(Y, X, Z, levels=5, colors = 'black')
    plt.clabel(XS, XS.levels, inline=True, fontsize=8)

    plt.contour(Y, X, conservative_approach_synth, levels=1, linestyles='dashed', colors='blue')
    plt.contour(Y, X, maximal_element_synth, levels=1, linestyles='dashed', colors='blue')
    plt.contour(Y, X, optimist_rule_synth, levels=1, linestyles='solid', colors='purple')
    # plt.show()

    plt.savefig('img/' + feature_names[0] + '-' + feature_names[1])
    plt.clf()