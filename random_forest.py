from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np

def random_forest_predict(features_train, features_test):
    y_train = features_train[:, -1]
    y_train = y_train.tolist()

    # remove the training labels from the dataset
    features_train = np.delete(features_train, -1, 1)

    y_test = features_test[:, -1]
    features_test = np.delete(features_test, -1, 1)

    model = RandomForestClassifier(random_state=0, class_weight="balanced").fit(features_train, y_train)
    predictions = cross_val_predict(model, features_test, y_test, cv=10)

    # probs = model.predict_proba(features_test)

    return predictions, model
