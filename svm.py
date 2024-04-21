import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
# from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt  # doctest: +SKIP
# from sklearn import metrics

def svm_predict(features_train, features_test, y_train):
    '''
    features_train <array>: The features and ground truth to train on.\n
    features_test <array>: The features to test on.\n

    Can accept any number of features as long as the label/ground truth is on the end of the array.\n
    Input must be a numpy array.\n

    Returns the predicted labels of the model.
    '''
    model = LinearSVC(random_state=0, tol=1e-5, max_iter=100)

    # fit the training dataset to the SVM classifier
    model.fit(features_train, y_train)

    y_test = features_test[:, -1]
    features_test = np.delete(features_test, -1, 1)

    # predict using the model trained
    svm_predict = cross_val_predict(model, features_test, y_test, cv=10)
    decision_function = model.decision_function(features_test)

    return svm_predict, model

def one_class(features_train, features_test):
    '''
    features_train <array>: The features and ground truth to train on.\n
    features_test <array>: The features to test on.\n

    Returns the predicted labels of the model given the one-class training.
    '''

    # create SVM classifier model
    model = OneClassSVM(gamma="scale", kernel="poly")

    # fit the training dataset to the SVM classifier
    model.fit(features_train)

    # predict using the model trained
    svm_predict = model.predict(features_test)

    return svm_predict


def svm_main(features_train, features_test):
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    
    y_train = features_train[:, -1]
    y_train = y_train.tolist()

    if 1 not in y_train or 0 not in y_train:
        # retrieve our training labels
        features_train = np.delete(features_train, -1, 1)

        # do SVM one-class training
        prediction = one_class(features_train, features_test)

        return prediction

    else:
        features_train = np.delete(features_train, -1, 1)

        # call svm_predict
        prediction = svm_predict(features_train, features_test, y_train)

        return prediction