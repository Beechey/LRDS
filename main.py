import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, plot_roc_curve, auc
from process import get_data
from process import clean_dataset
from lrds import calculate_weights
from lrds import calculate_positive_negative_parts
from lrds import calculate_mass_functions
from lrds import sum_weights
from lrds import join_mass_functions
from lrds import calculate_plausibility
from lrds import calculate_belief
from lrds import decision_rule
from lrds import get_interval
from lrds import remove_contradictory_instances
from lrds import create_figure_7
from lrds import create_figure_8
from feature_selection import calculate_average_mass
from feature_selection import attach_feature_names
from feature_selection import sort_dict
from feature_selection import output_feature_ranking
from feature_selection import measure_feature_correctness
from tabulate_results import id_rule_decision
from tabulate_results import remove_old_results
from tabulate_results import create_cm
from bayes import bayes_predict
from svm import svm_main
from decision_tree import tree_predict
from random_forest import random_forest_predict

# Get data from dataset - nX, rX, Y: normalised, raw and labels
nX, rX, Y, feature_names, technique = get_data()

test_size = input("\nPlease input a decimal value for the size of the testing dataset (0.35 = 35%): ")
test_size = float(test_size)

run_type = input("Please input 1 if ranking features, 2 if running experiments, 3 if generating a figure 7, or 4 for figure 8s: ")
run_type = int(run_type)

if run_type != 1 and run_type != 2 and run_type!= 3 and run_type != 4:
    print("Error: You must select a valid run type!")
    sys.exit()

if test_size == 0.0:
    print("Error: You must have a testing dataset!")
    sys.exit()
elif test_size == 1.0:
    nX_train = nX
    nX_test = nX # nX = normalised data
    y_train = Y
    y_test = Y

    rX_train = rX # rX = raw data
    rX_test = rX

    nX_train_LR, nX_train_ML, y_train_LR, y_train_ML = train_test_split(nX_train, y_train, test_size=0.5, random_state=75)
    rX_train_LR, rX_train_ML, y_train_LR, y_train_ML = train_test_split(rX_train, y_train, test_size=0.5, random_state=75)
else:
    print("\nSplitting the dataset... ", end="", flush=True)
    nX_train, nX_test, y_train, y_test = train_test_split(nX, Y, test_size=test_size, random_state=75)
    rX_train, rX_test, y_train, y_test = train_test_split(rX, Y, test_size=test_size, random_state=75)

    nX_train_LR, nX_train_ML, y_train_LR, y_train_ML = train_test_split(nX_train, y_train, test_size=0.5, random_state=75)
    rX_train_LR, rX_train_ML, y_train_LR, y_train_ML = train_test_split(rX_train, y_train, test_size=0.5, random_state=75)
    print("Done!")

dataset_test_size = test_size * 100
dataset_test_size = str(int(dataset_test_size))

D = nX.shape[1] # number of features
W = np.random.randn(D) # initialise feature weights (parameters)
b = 0 # initialise bias

remove_old_results(technique)

print("\nTraining Logistic Regression... ", end="", flush=True)
start = time.process_time()
lr_model = LogisticRegression(solver="liblinear", random_state=0, max_iter=100, class_weight="balanced").fit(nX_train_LR, y_train_LR)
time_lr_train = time.process_time() - start
print("Done!")

W = lr_model.coef_[0] # feature weights (parameters)
b = lr_model.intercept_[0] # bias
alpha = b/D
print("Calculating weights... ", end="", flush=True)
weights = calculate_weights(b, nX_test, W, alpha)
print("Done!")

print("Calculating positive and negative parts... ", end="", flush=True)
positive_part, negative_part = calculate_positive_negative_parts(weights)
print("Done!")

if run_type == 1: # if we are ranking features
    print("Calculating mass functions... ", end="", flush=True)
    theta1, theta2, theta, k = calculate_mass_functions(positive_part, negative_part)
    print("Done!")

    print("Measuring correctness of each feature... ", end="", flush=True)
    theta1_correct_classifications = measure_feature_correctness(theta1, y_test)
    print("Done!")

    print("Ranking features by validity of decisions... ", end="", flush=True)
    theta1_feature_dict = attach_feature_names(feature_names, theta1_correct_classifications)
    sorted_theta1 = sort_dict(theta1_feature_dict, True)
    output_feature_ranking(sorted_theta1, "Validity", "validity")
    print("Done!")

    print("Calculating averages and medians... ", end="", flush=True)
    theta_averages, theta_median = calculate_average_mass(theta)
    k_averages, k_median = calculate_average_mass(k)

    conflict_and_uncertainty = theta_averages + k_averages
    print("Done!")

    print("Ranking features by uncertainty... ", end="", flush=True)
    theta_feature_average_dict = attach_feature_names(feature_names, theta_averages)
    sorted_theta_average = sort_dict(theta_feature_average_dict)
    output_feature_ranking(sorted_theta_average, "Uncertainty", "mean")

    theta_feature_median_dict = attach_feature_names(feature_names, theta_median)
    sorted_theta_median = sort_dict(theta_feature_median_dict)
    output_feature_ranking(sorted_theta_median, "Uncertainty", "median")
    print("Done!")
elif run_type == 2: # we are running experiments
    print("Summing the positive and negative parts... ", end="", flush=True)
    positive_part, negative_part = sum_weights(positive_part, negative_part)
    print("Done!")

    print("Calculating mass functions... ", end="", flush=True)
    theta1, theta2, theta, k = calculate_mass_functions(positive_part, negative_part) # <--- we get division by 0 errors  (x/1-k where k == 1)
    y_test_reduced = y_test
    print("Done!")

    print("Joining mass functions... ", end="", flush=True)
    joined_mass_functions = join_mass_functions(theta1, theta2, theta)
    print("Done!")

    print("Calculating belief... ", end="", flush=True)
    theta1_bel = calculate_belief(joined_mass_functions, "1")
    theta2_bel = calculate_belief(joined_mass_functions, "2")
    print("Done!")

    print("Calculating plausibility... ", end="", flush=True)
    theta1_pl = calculate_plausibility(joined_mass_functions, "1")
    theta2_pl = calculate_plausibility(joined_mass_functions, "2")
    print("Done!")

    print("Performing Pessimist Rule... ", end="", flush=True)
    pessimist_rule = decision_rule(theta1_bel, theta2_bel, "pessimist")
    print("Done!")

    print("Calculating MP Rule... ", end="", flush=True)
    start = time.process_time()
    mp_rule = decision_rule(theta1_pl, theta2_pl, "optimist") # Optimist Rule
    time_mp_rule = time.process_time() - start
    print("Done!")

    print("Performing Conservative Approach... ", end="", flush=True)
    conservative_approach = decision_rule(theta1_bel, theta2_pl, "conservative")
    print("Done!")

    print("Performing Maximal Element... ", end="", flush=True)
    maximal_element = decision_rule(theta1_pl, theta2_bel, "maximal")
    print("Done!")

    print("Calculating ID Rule... ", end="", flush=True)
    start = time.process_time()
    id_rule = get_interval(conservative_approach, maximal_element)
    time_id_rule = time.process_time() - start
    print("Done!")

    print("Testing Logistic Regression (raw features)... ", end="", flush=True)
    start = time.process_time()
    lr_predictions = cross_val_predict(lr_model, nX_test, y_test, cv=10)
    # lr_predictions = lr_model.predict(nX_test)
    lr_probs = lr_model.predict_proba(nX_test)
    time_lr_test = time.process_time() - start
    time_lr_total = time_lr_train + time_lr_test
    print("Done!")

    # Train and test different ML algorithms
    print("Training and testing Naive Bayes (raw features)... ", end="", flush=True)
    start = time.process_time()
    bayes_features_train = np.column_stack((nX_train_ML, y_train_ML))
    bayes_features_test = np.column_stack((nX_test, y_test))
    bayes_predicted, bayes_model = bayes_predict(bayes_features_train, bayes_features_test)
    time_bayes = time.process_time() - start
    print("Done!")

    print("Training and testing SVM (raw features)... ", end="", flush=True)
    start = time.process_time()
    svm_features_train = np.column_stack((nX_train_ML, y_train_ML))
    svm_features_test = np.column_stack((nX_test, y_test))
    svm_predicted, svm_model = svm_main(svm_features_train, svm_features_test)
    svm_predicted = np.where(svm_predicted == -1, 0, svm_predicted)  # replace the -1 values with 0 to make analysis easier
    time_svm = time.process_time() - start
    print("Done!")

    print("Training and testing Decision Tree (raw features)... ", end="", flush=True)
    start = time.process_time()
    decision_tree_features_train = np.column_stack((nX_train_ML, y_train_ML))
    decision_tree_features_test = np.column_stack((nX_test, y_test))
    decision_tree_predicted, decision_tree_model = tree_predict(decision_tree_features_train, decision_tree_features_test)
    time_decision_tree = time.process_time() - start
    print("Done!")

    print("Training and testing Random Forest (raw features)... ", end="", flush=True)
    start = time.process_time()
    random_forest_features_train = np.column_stack((nX_train_ML, y_train_ML))
    random_forest_features_test = np.column_stack((nX_test, y_test))
    random_forest_predicted, random_forest_model = tree_predict(random_forest_features_train, random_forest_features_test)
    time_random_forest = time.process_time() - start
    print("Done!")

    print("Creating ROC Curves (raw features)... ", end="", flush=True)
    fpr, tpr, _ = roc_curve(y_test_reduced, theta1)
    roc_auc = auc(fpr, tpr)

    bayes_features_test = np.delete(bayes_features_test, -1, 1)
    svm_features_test = np.delete(svm_features_test, -1, 1)
    decision_tree_features_test = np.delete(decision_tree_features_test, -1, 1)
    random_forest_features_test = np.delete(random_forest_features_test, -1, 1)

    bayes_roc = plot_roc_curve(bayes_model, bayes_features_test, y_test_reduced, name="Naive Bayes")     
    lr_roc = plot_roc_curve(lr_model, nX_test, y_test_reduced, name="Logistic Regression", ax=bayes_roc.ax_)  
    svm_roc = plot_roc_curve(svm_model, svm_features_test, y_test_reduced, name="Support Vector Machine", ax=lr_roc.ax_)  
    dt_roc = plot_roc_curve(decision_tree_model, decision_tree_features_test, y_test_reduced, name="Decision Tree", ax=svm_roc.ax_)  
    rf_roc = plot_roc_curve(random_forest_model, random_forest_features_test, y_test_reduced, name="Random Forest", ax=dt_roc.ax_) 

    plt.plot(fpr, tpr, linestyle='--', label = 'LR-DS (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.legend(loc="lower right")

    plt.savefig("results/" + technique + "/" + technique + "_raw_ROC" + ".svg", format='svg', dpi=1200)
    print("Done!")

    create_cm(lr_predictions, y_test_reduced, "lr", time_lr_total, time_lr_train, technique)
    create_cm(mp_rule, y_test_reduced, "mp_rule", time_mp_rule, time_lr_train, technique)
    id_rule_decision(id_rule, y_test_reduced, time_id_rule, time_lr_train, technique)
    create_cm(bayes_predicted, y_test_reduced, "bayes", time_bayes, time_lr_train, technique)
    create_cm(svm_predicted, y_test_reduced, "svm", time_svm, time_lr_train, technique)
    create_cm(decision_tree_predicted, y_test_reduced, "decision_tree", time_decision_tree, time_lr_train, technique)
    create_cm(random_forest_predicted, y_test_reduced, "random_forest", time_random_forest, time_lr_train, technique)

elif run_type == 3: # figure 7s
    print("Calculating mass functions... ", end="", flush=True)
    theta1, theta2, theta, k = calculate_mass_functions(positive_part, negative_part)
    print("Done!")

    np.savetxt("theta1.csv", theta1, delimiter=",")
    np.savetxt("theta2.csv", theta2, delimiter=",")
    np.savetxt("theta.csv",  theta, delimiter=",")

    np.savetxt("rX.csv", rX_test, delimiter=",")

    for i in range(D):
        create_figure_7(rX_test[:, i], theta1[:, i], theta2[:, i], theta[:, i], feature_names[i])

elif run_type == 4: # figure 8s
    feature_couples = []
    W = W.tolist()

    for i in range(0, D):
        for j in range(i+1, D):
            feature_couples.append([str(i), str(j)])

    for i in feature_couples:
        feature1 = int(i[0])
        feature2 = int(i[1])

        feature_name_couple = [feature_names[feature1], feature_names[feature2]]
        normalised_features = nX_train_LR[:, [feature1, feature2]]
        
        print("\nTraining Logistic Regression for " + feature_names[feature1] + " and " + feature_names[feature2] + "... ", end="", flush=True)
        lr_model = LogisticRegression(solver="liblinear", random_state=0, max_iter=100, class_weight="balanced").fit(normalised_features, y_train_LR)
        print("Done!")

        D = len(i)
        W = lr_model.coef_[0] # feature weights (parameters)
        b = lr_model.intercept_[0] # bias
        alpha = b/D

        print("weights: ", W)
        print("bias: ", b)
        print("alpha: ", alpha)

        W_couple = np.array(W)
        data = rX_test[:, [feature1, feature2]]
        
        feature_name_couple = [feature_names[feature1], feature_names[feature2]]
        feature_name_couple = np.char.replace(feature_name_couple, '/', '')

        print("Creating figure 8 for " + feature_names[feature1] + " and " + feature_names[feature2] + "... ", end="", flush=True)
        create_figure_8(W_couple, b, D, data, feature_name_couple, y_test)
        print("Done!")