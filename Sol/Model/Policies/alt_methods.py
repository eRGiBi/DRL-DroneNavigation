import os
import itertools

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def read_data():

    rollouts = pd.DataFrame()

    for file in os.listdir(".\Sol/rollouts"):
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

def svm(x_train, y_train, x_test, y_test):

    svm_classifier = SVC(kernel='poly')

    svm_classifier.fit(x_train, y_train)

    return accuracy_score(y_test, svm_classifier.predict(x_test))

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


def radiusnr (x_train, y_train, x_test, y_test):

    radiusnr_classifier = RadiusNeighborsClassifier()

    radiusnr_classifier.fit(x_train, y_train )

    y_pred = radiusnr_classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)

def linear_regression(x_train, y_train, x_test, y_test):

    X_train_norm = x_train
    X_test_norm = x_test

    # print(f'Before normalizing:\nMax value: {max(x_train)}\nMin value: {min(x_train)}')
    # scaler = MinMaxScaler()
    # scaler.fit(x_train)
    # X_train_norm = scaler.transform(x_train)
    # X_test_norm = scaler.transform(x_test)
    # print(f'After normalizing:\nMax value: {max(X_train_norm)}\nMin value: {min(X_train_norm)}')

    regressor = LinearRegression()
    regressor.fit(X_train_norm, y_train)

    y_pred = regressor.predict(X_test_norm)

    print("Linear Regression Results: -------------------")
    print(f'Predicted values: {y_pred[:5]}')
    print(f'Actual values: {y_test[:5]}')

    R_squared = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    MAE = mean_absolute_error(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)

    print(f'R-squared: {round(R_squared * 100, 2)}%')
    print(f'Mean Absolute Error (MAE): {round(MAE, 2)}')
    print(f'Mean Squared Error (MSE): {round(MSE, 2)}')
    print(f'Root Mean Squared Error (RMSE): {round(RMSE, 2)}')
    print(f'Mean Absolute Percentage Error (MAPE): {round(MAPE, 2)}%')
    print()

def ridge_reg(x_train, x_test, y_train, y_test):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    alphas = [0.1, 1.0, 10.0]

    print("Ridge Regression Results: -------------------")

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)

        scores = cross_val_score(ridge, x_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

        mean_score = np.mean(scores)

        print(f"Alpha: {alpha}, Mean Squared Error: {mean_score}")

        ridge.fit(x_train_scaled, y_train)

        y_pred = ridge.predict(x_test_scaled)

        R_squared = r2_score(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = mean_squared_error(y_test, y_pred, squared=False)
        MAE = mean_absolute_error(y_test, y_pred)
        MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

        print(f"For Alpha: {alpha}")
        print(f'R-squared: {round(R_squared * 100, 2)}%')
        print(f'Mean Absolute Error (MAE): {round(MAE, 2)}')
        print(f'Mean Squared Error (MSE): {round(MSE, 2)}')
        print(f'Root Mean Squared Error (RMSE): {round(RMSE, 2)}')
        print(f'Mean Absolute Percentage Error (MAPE): {round(MAPE, 2)}%')
        print()

def lasso_reg(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Define different alpha values to try
    alphas = [0.1, 1.0, 10.0, ]

    print("Lasso Regression Results: -------------------")

    for alpha in alphas:
        # Initialize Lasso regression model
        lasso = Lasso(alpha=alpha)

        # Fit the model
        lasso.fit(x_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = lasso.predict(x_test_scaled)

        # Calculate evaluation metrics
        R_squared = r2_score(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = mean_squared_error(y_test, y_pred, squared=False)
        MAE = mean_absolute_error(y_test, y_pred)
        MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

        # Print evaluation metrics with their full names
        print(f"For Alpha: {alpha}")
        print(f'R-squared: {round(R_squared * 100, 2)}%')
        print(f'Mean Absolute Error (MAE): {round(MAE, 2)}')
        print(f'Mean Squared Error (MSE): {round(MSE, 2)}')
        print(f'Root Mean Squared Error (RMSE): {round(RMSE, 2)}')
        print(f'Mean Absolute Percentage Error (MAPE): {round(MAPE, 2)}%')
        print()

def poly_reg(x_train, x_test, y_train, y_test):

    degrees = [1,  3, 4, ]

    print("Lasso Regression Results: -------------------")

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)

        poly_regression = make_pipeline(poly_features, LinearRegression())

        y_pred_cv = cross_val_predict(poly_regression, x_train, y_train, cv=5)

        R_squared = r2_score(y_train, y_pred_cv)
        MSE = mean_squared_error(y_train, y_pred_cv)
        RMSE = mean_squared_error(y_train, y_pred_cv, squared=False)
        MAE = mean_absolute_error(y_train, y_pred_cv)
        MAPE = np.mean(np.abs((y_train - y_pred_cv) / y_train)) * 100  # Mean Absolute Percentage Error

        print(f'For Degree of Polynomial: {degree}')
        print(f'R-squared: {round(R_squared * 100, 2)}%')
        print(f'Mean Absolute Error (MAE): {round(MAE, 2)}')
        print(f'Mean Squared Error (MSE): {round(MSE, 2)}')
        print(f'Root Mean Squared Error (RMSE): {round(RMSE, 2)}')
        print(f'Mean Absolute Percentage Error (MAPE): {round(MAPE, 2)}%')
        print()

if __name__ == "__main__":

    x_train, x_test, y_train, y_test = read_data()
    linear_regression(x_train, y_train, x_test, y_test)
    ridge_reg(x_train, x_test, y_train, y_test)
    lasso_reg(x_train, x_test, y_train, y_test)
    poly_reg(x_train, x_test, y_train, y_test)

    #
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

    radiusnr_acc = radiusnr(x_train, y_train, x_test, y_test)
    print(f"Radius Neighbors accuracy: {radiusnr_acc}")


