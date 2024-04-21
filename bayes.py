import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict

def bayes_predict(features_train, features_test):
    '''
    features_train <array>: The features and ground truth to train on.\n
    features_test <array>: The features to test on.\n

    Can accept any number of features as long as the label/ground truth is on the end of the array.\n
    Input must be a numpy array.\n

    Returns the predicted labels of the model.
    '''

    # retrieve our training labels
    y_train = features_train[:, -1]
    y_train = y_train.tolist()

    y_test = features_test[:, -1]

    # remove the training labels from the dataset
    features_train = np.delete(features_train, -1, 1)
    features_test = np.delete(features_test, -1, 1)

    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    # create Naive Bayes model
    model = GaussianNB()

    # train Naive Bayes model
    model.fit(features_train, y_train)

    # predict using Naive Bayes
    bayes_predicted = cross_val_predict(model, features_test, y_test, cv=10)

    # bayes_predicted = model.predict(features_test)
    # bayes_probs = model.predict_proba(features_test)

    return bayes_predicted, model
    # return bayes_predicted, bayes_probs