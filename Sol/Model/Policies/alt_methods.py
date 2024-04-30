import os
import itertools

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier


def read_data():

    rollouts = pd.DataFrame()

    for file in os.listdir("Sol/rollouts"):
        print(file)
        # if file.endswith(".csv"):
        #     rollouts = pd.concat([rollouts, pd.read_csv(f"Sol/rollouts/{file}")], ignore_index=True)
        # if file.endswith(".gz"):
        #     rollouts = pd.concat([rollouts, pd.read_csv(f"Sol/rollouts/{file}", compression='gzip')], ignore_index=True)
        if file.endswith(".txt"):
            rollouts = pd.concat([rollouts, pd.read_table(f"Sol/rollouts/{file}", sep=',', header=None)], ignore_index=True)

        # rollouts = np.loadtxt(f"Sol/rollouts/{file}", delimiter=',')

    rollouts = rollouts.dropna()
    rollouts = rollouts.to_numpy()

    x = rollouts[:, :-1]
    y = rollouts[:, -1]
    print(x.shape, y.shape)
    print(x[0], y[0])
    print(type(x[0]), type(y[0]))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = x_train.tolist()
    x_test = x_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    # for i in range(len(x_train)):
    #     # x_train[i] = x_train[i].tolist()
    #     for j in range(len(x_train[i])):
    #         x_train[i][j] = getattr(x_train[i][j], "tolist", lambda: x_train[i][j])()
    # print(x_train[0], type(x_train[0][0]))


    # for z in itertools.chain(x_train, x_test):
    #     for i in z:
    #         print(i, type(i))
    #         i = getattr(i, "tolist", lambda: i)()
    #         print(i, type(i))
    # for i in y_train:
    #     i = getattr(i, "tolist", lambda: i)()
    # for i in y_test:
    #     i = getattr(i, "tolist", lambda: i)()

    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(x_train[0], y_train[0])

    print(type(x_train[0][0]), type(y_train[0]))


    return x_train, x_test, y_train, y_test


# SVM
def svm_param_search(X_train, y_train, X_test, y_test):

    clf = SVC()

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel types
        'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf' and 'poly'
        'degree': [2, 3, 4]  # Degree of the polynomial kernel
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Best parameters found
    print("Best parameters:", grid_search.best_params_)
    print(grid_search.best_score_)

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    return acc

def svm(x_train, y_train, x_test, y_test):

    svm_classifier = SVC(kernel='poly')

    svm_classifier.fit(x_train, y_train)

    return accuracy_score(y_test, svm_classifier.predict(x_test))

def naive_bayes(x_train, y_train, x_test, y_test):

    naive_bayes_classifier = GaussianNB()

    naive_bayes_classifier.fit(x_train, y_train)

    y_pred = naive_bayes_classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)

def naive_bayes_search(x_train, y_train, x_test, y_test):

    pipeline = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )
    param_grid = {
        'gaussiannb__var_smoothing': [1e-9, 1e-8, 1e-7]  # Smoothing parameter
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    print("Best NB parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    return accuracy_score(y_test, y_pred)

def decision_tree(x_train, y_train, x_test, y_test):

    decision_tree_classifier = DecisionTreeClassifier()

    decision_tree_classifier.fit(x_train, y_train)

    y_pred = decision_tree_classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)

def radiusnr (x_train, y_train, x_test, y_test):

    radiusnr_classifier = RadiusNeighborsClassifier()

    radiusnr_classifier.fit(x_train, y_train )

    y_pred = radiusnr_classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":

    x_train, x_test, y_train, y_test = read_data()



    # svm_acc = svm(x_train, y_train, x_test, y_test)
    #
    # print(f"Accuracy: {svm_acc}")

    # optim_svm = svm_param_search(x_train, y_train, x_test, y_test)
    #
    # print(f"Optimized SVM accuracy: {optim_svm}")
    #
    #
    # naive_bayes_acc = naive_bayes(x_train, y_train, x_test, y_test)
    #
    # print(f"Naive Bayes accuracy: {naive_bayes_acc}")
    #

    decision_tree_acc = decision_tree(x_train, y_train, x_test, y_test)

    print(f"Decision Tree accuracy: {decision_tree_acc}")



    print(f"Radius Neighbors accuracy: {radiusnr_acc}")


