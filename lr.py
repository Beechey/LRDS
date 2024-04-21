import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from process import get_data


# Get data from dataset - nX, rX, Y: normalised, raw and labels
nX, rX, Y, feature_names = get_data()

# Make copies for rX, nX, Y because numpy points to object, rather than copies content, when using assignment
unshuffled_rX = rX
unshuffled_nX = nX
unshuffled_Y = Y

test_size = input("Please input a decimal value for the size of the testing dataset (0.3 = 30%): ")
test_size = float(test_size)

if test_size == 1.0:
    nX_train = nX
    nX_test = nX
    y_train = Y
    y_test = Y

    rX_train = rX
    rX_test = rX

    nX_train_LR, nX_train_SVM, y_train_LR, y_train_SVM = train_test_split(nX_train, y_train, test_size=0.5, random_state=42)
    rX_train_LR, rX_train_SVM, y_train_LR, y_train_SVM = train_test_split(rX_train, y_train, test_size=0.5, random_state=42)
else:
    nX_train, nX_test, y_train, y_test = train_test_split(nX, Y, test_size=test_size, random_state=42)
    rX_train, rX_test, y_train, y_test = train_test_split(rX, Y, test_size=test_size, random_state=42)

    nX_train_LR, nX_train_SVM, y_train_LR, y_train_SVM = train_test_split(nX_train, y_train, test_size=0.5, random_state=42)
    rX_train_LR, rX_train_SVM, y_train_LR, y_train_SVM = train_test_split(rX_train, y_train, test_size=0.5, random_state=42)

dataset_test_size = test_size * 100
dataset_test_size = str(int(dataset_test_size))

y_train_LR = y_train_LR.tolist()
y_test = y_test.tolist()

clf = LogisticRegression(solver="liblinear", penalty="none", random_state=0, max_iter=100, verbose=1).fit(nX_train_LR, y_train_LR)
print("\n Coefs: ", clf.coef_)

predictions = clf.predict(rX_test)

score = clf.score(rX_test, y_test)
print("Accuracy: ", score)

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print("tp: ", tp)
print("fp: ", fp)
print("fn: ", fn)
print("tn: ", tn)