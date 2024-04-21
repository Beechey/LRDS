import warnings
import os
import numpy as np
from tabulate import tabulate
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def remove_old_results(technique):
    path = "results/" + technique + "/"

    methods = [
        "id_rule",
        "id_rule_split",
        "mp_rule",
        "lr",
        "bayes",
        "svm",
        "decision_tree",
        "random_forest"
    ]

    try:
        os.mkdir(path)
        print("Creating directory " + path)
    except:
        print("Path " + path + " already exists.")

    for method in methods:
        try:
            os.remove(path + method + "_cm.txt")
            print("File " + path + method + "_cm.txt deleted.")
        except IOError:
            print("File " + path + method + "_cm.txt not accessible.")

    try:
        os.mkdir("full_results/")
        print("Creating directory " + "full_results/")
    except:
        print("Path " + "full_results/" + " already exists.")

    try:
        os.remove("full_results/" + technique + "_results.csv")
        print("File full_results/" + technique + "_results.csv deleted.")
    except:
        print("File full_results/" + technique + "_results.csv not accessible.")

def confusion_values(prediction, y):
    '''
    Calculates the confusion matrix based from the prediction and labels presented.\n
    '''

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for pred, label in zip(prediction, y):
        if label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 1 and pred == 1:
            tp += 1

    return tn, fp, fn, tp

def calculate_precision(pred, y):
    warnings.filterwarnings('ignore')
    p = precision_score(y, pred)
    return p

def calculate_recall(pred, y):
    warnings.filterwarnings('ignore')
    r = recall_score(y, pred)
    return r

def calculate_f1_score(pred, y):
    warnings.filterwarnings('ignore')
    f1 = f1_score(y, pred)
    return f1

def create_cm(predictions, y, algorithm, time, initial_time, technique):
    '''
    predictions <array>: The predictions made through Logistic Regression
    y <array>: The ground truth from the dataset

    Creates a confusiuon matrix, writes it into a table and then writes that table to an external file (lr_cm.txt).
    '''

    try:
        y = y.tolist()
    except:
        pass

    try:
        predictions = predictions.tolist()
    except:
        pass

    tn, fp, fn, tp = confusion_values(predictions, y)

    precision = calculate_precision(predictions, y)
    recall = calculate_recall(predictions, y)
    f1_score = calculate_f1_score(predictions, y)

    total_num = tn + fp + fn + tp
    algorithm = str(algorithm)

    tp_perc = (tp / total_num) * 100
    fp_perc = (fp / total_num) * 100

    fn_perc = (fn / total_num) * 100
    tn_perc = (tn / total_num) * 100

    row_names = [
        'True Positives',
        'False Positives',
        'False Negatives',
        'True Negatives',
        'Precision Score',
        'Recall Score',
        'F1 Score',
        'Initial time',
        'Time taken'
    ]

    column_data = [
        tp_perc,
        fp_perc,
        fn_perc,
        tn_perc,
        str(np.round(precision, 2)),
        str(np.round(recall, 2)),
        str(np.round(f1_score, 2)),
        initial_time,
        time
    ]

    new_column = pd.DataFrame()
    new_column[algorithm] = column_data

    if os.path.exists("full_results/" + technique + "_results.csv") == True:
        df_csv = pd.read_csv("full_results/" + technique + "_results.csv")
        df_csv = pd.concat([df_csv, new_column], axis=1)
        df_csv.to_csv("full_results/" + technique + "_results.csv", index=False)

    else:
        new_column.index = row_names
        new_column.to_csv("full_results/" + technique + "_results.csv")


    cm_file = open("results/" + technique + "/" + algorithm +  "_cm.txt", "a")
    cm_data = []

    cm_data.append("Initial LR time: " + str(initial_time) + "\n")
    cm_data.append("Time Taken: " + str(time) + "\n")

    cm_data.append("\nPrecision: " + str(np.round(precision, 2)) + "\n")
    cm_data.append("Recall: " + str(np.round(recall, 2)) + "\n")
    cm_data.append("F1 Score: " + str(np.round(f1_score, 2)) + "\n\n")

    cm_text = [
        ["", "Positive", "Negative"],
        ["Positive", np.round(tp_perc, 2), np.round(fp_perc, 2)],
        ["Negative", np.round(fn_perc, 2), np.round(tn_perc, 2)],
    ]

    cm_file.writelines(cm_data)
    cm_file.write(tabulate(cm_text))
    cm_file.close()

def id_rule_decision(interval, y_test, time, initial_time, technique):
    interval = interval.tolist()

    tp = []
    fp = []
    fn = []
    tn = []

    ut = []
    uf = []

    for id, y in zip(interval, y_test):
        if id == 1 and y == 1:
            tp.append(1)
            fp.append(0)
            fn.append(0)
            tn.append(0)
            ut.append(0)
            uf.append(0)
        elif id == 1 and y == 0:
            tp.append(0)
            fp.append(1)
            fn.append(0)
            tn.append(0)
            ut.append(0)
            uf.append(0)
        elif id == 0 and y == 1:
            tp.append(0)
            fp.append(0)
            fn.append(1)
            tn.append(0)
            ut.append(0)
            uf.append(0)
        elif id == 0 and y == 0:
            tp.append(0)
            fp.append(0)
            fn.append(0)
            tn.append(1)
            ut.append(0)
            uf.append(0)
        elif id == 2 and y == 1:
            tp.append(0)
            fp.append(0)
            fn.append(0)
            tn.append(0)
            ut.append(1)
            uf.append(0)
        elif id == 2 and y == 0:
            tp.append(0)
            fp.append(0)
            fn.append(0)
            tn.append(0)
            ut.append(0)
            uf.append(1)

    tp = tp.count(1)
    fp = fp.count(1)
    fn = fn.count(1)
    tn = tn.count(1)
    ut = ut.count(1)
    uf = uf.count(1)

    total_num = len(y_test)

    tp_perc = (tp / total_num) * 100
    fp_perc = (fp / total_num) * 100
    u_pos_perc = (ut / total_num) * 100

    fn_perc = (fn / total_num) * 100
    tn_perc = (tn / total_num) * 100
    u_neg_perc = (uf / total_num) * 100

    tp_perc_u = tp_perc + (u_pos_perc/2)
    fp_perc_u = fp_perc + (u_neg_perc/2)
    fn_perc_u = fn_perc + (u_pos_perc/2)
    tn_perc_u = tn_perc + (u_neg_perc/2)
    
    precision = tp_perc_u / (tp_perc_u + fp_perc_u)
    recall = tp_perc_u / (tp_perc_u + fn_perc_u)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    precision_u = tp_perc_u / (tp_perc_u + fp_perc_u)
    recall_u = tp_perc_u / (tp_perc_u + fn_perc_u)
    f1_score_u = 2 * ((precision_u * recall_u) / (precision_u + recall_u))

    row_names = [
        'True Positives',
        'False Positives',
        'False Negatives',
        'True Negatives',
        'Precision Score',
        'Recall Score',
        'F1 Score',
        'Initial time',
        'Time taken'
    ]

    column_data = [
        tp_perc,
        fp_perc,
        fn_perc,
        tn_perc,
        str(np.round(precision_u, 2)),
        str(np.round(recall_u, 2)),
        str(np.round(f1_score_u, 2)),
        initial_time,
        time
    ]

    new_column = pd.DataFrame()
    new_column["id_rule"] = column_data

    if os.path.exists("full_results/" + technique + "_results.csv") == True:
        df_csv = pd.read_csv("full_results/" + technique + "_results.csv")
        df_csv = pd.concat([df_csv, new_column], axis=1)
        df_csv.to_csv("full_results/" + technique + "_results.csv", index=False)

    else:
        new_column.index = row_names
        new_column.to_csv("full_results/" + technique + "_results.csv")


    cm_file = open("results/" + technique + "/id_rule_split_cm.txt", "a")

    cm_data = []

    cm_data.append("Initial LR time: " + str(initial_time) + "\n")
    cm_data.append("Time Taken: " + str(time) + "\n")

    cm_data.append("\nPrecision: " + str(np.round(precision, 2)) + "\n")
    cm_data.append("Recall: " + str(np.round(recall, 2)) + "\n")
    cm_data.append("F1 Score: " + str(np.round(f1_score, 2)) + "\n\n")

    confusion_matrix = [
        ["", "Positive", "Negative"],
        ["Positive", np.round(tp_perc_u, 2), np.round(fp_perc_u, 2)],
        ["Negative", np.round(fn_perc_u, 2), np.round(tn_perc_u, 2)],
    ]

    cm_file.writelines(cm_data)
    cm_file.write(tabulate(confusion_matrix))
    cm_file.close()

    try:
        precision = tp_perc / (tp_perc + fp_perc)
    except:
        precision = tp_perc / ((tp_perc + fp_perc) + 0.00000001)

    try:
        recall = tp_perc / (tp_perc + fn_perc)
    except:
        recall = tp_perc / ((tp_perc + fn_perc) + 0.00000001)

    print("p: ", precision)
    print("r: ", recall)
    
    try:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    except:
        f1_score = 2 * ((precision * recall) / (precision + recall + 0.0000000001))

    cm_file = open("results/" + technique + "/id_rule_cm.txt", "a")

    cm_data = []

    cm_data.append("Initial LR time: " + str(initial_time) + "\n")
    cm_data.append("Time Taken: " + str(time) + "\n")

    cm_data.append("\nPrecision: " + str(np.round(precision, 2)) + "\n")
    cm_data.append("Recall: " + str(np.round(recall, 2)) + "\n")
    cm_data.append("F1 Score: " + str(np.round(f1_score, 2)) + "\n\n")

    confusion_matrix = [
        ["", "Positive", "Negative"],
        ["Positive", np.round(tp_perc, 2), np.round(fp_perc, 2)],
        ["Negative", np.round(fn_perc, 2), np.round(tn_perc, 2)],
        ["Uncertain", np.round(u_pos_perc, 2), np.round(u_neg_perc, 2)]
    ]

    cm_file.writelines(cm_data)
    cm_file.write(tabulate(confusion_matrix))
    cm_file.close()    