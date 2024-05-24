import os
import itertools
import time
from functools import wraps

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
# from matplotlib.backend_managers.ToolManager import tools
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors._regression import KNeighborsRegressor

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict

from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score, normalized_mutual_info_score, \
    fowlkes_mallows_score, silhouette_score
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor, export_graphviz


def filter_lines_by_length(file_path, length):
    filtered_lines = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().split(',')
                if len(line) == length:
                    filtered_lines.append(line)
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return filtered_lines


def read_data():
    rollouts = pd.DataFrame()

    for file in os.listdir(".\Sol/rollouts"):
        print(file)
        if file.endswith(".csv"):
            rollouts = pd.concat([rollouts, pd.read_csv(f"Sol/rollouts/{file}")], ignore_index=True)
        if file.endswith(".gz"):
            rollouts = pd.concat([rollouts, pd.read_csv(f"Sol/rollouts/{file}", compression='gzip')], ignore_index=True)
        if file.endswith(".txt"):

            filtered_lines = filter_lines_by_length(f"Sol/rollouts/{file}", 13)
            if filtered_lines:
                filtered_df = pd.DataFrame([line for line in filtered_lines])
                rollouts = pd.concat([rollouts, filtered_df], ignore_index=True)
            # rollouts = pd.concat([rollouts, pd.read_table(f"Sol/rollouts/{file}", sep=',', header=None)], ignore_index=True)

        # rollouts = np.loadtxt(f"Sol/rollouts/{file}", delimiter=',')

    print("Rollouts DataFrame after processing all files:")
    print(rollouts.head())

    rollouts = rollouts.dropna()
    rollouts = rollouts.to_numpy(dtype=float)

    x = rollouts[:, :-1]
    y = rollouts[:, -1]

    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"x[0]: {x[0]}, y[0]: {y[0]}")
    print(f"type(x[0]): {type(x[0])}, type(y[0]): {type(y[0])}")

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

    print(f"type(x_train[0]): {type(x_train[0])}, type(y_train[0]): {type(y_train[0])}")

    # print(type(x_train[0][0]), type(y_train[0]))

    return x_train, x_test, y_train, y_test


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        print()
        return result

    return wrapper


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


@measure_time
def KNeighbors(x_train, y_train, x_test, y_test):
    KNeighborsReg = KNeighborsRegressor()

    KNeighborsReg.fit(x_train, y_train)

    y_pred = KNeighborsReg.predict(x_test)

    return accuracy_score(y_test, y_pred)


@measure_time
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


@measure_time
def ridge_reg(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    alphas = [0.1, 1.0, 10.0]

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)

        ridge.fit(x_train_scaled, y_train)

        y_pred = ridge.predict(x_test_scaled)

        print(f"Ridge Regression Results for {alpha}: -------------------")
        print(f'Predicted values: {y_pred[:5]}')
        print(f'Actual values: {y_test[:5]}')

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


@measure_time
def lasso_reg(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Define different alpha values to try
    alphas = [0.1, 1.0, 10.0, ]

    for alpha in alphas:
        print(f"Lasso Regression Results for {alpha}: -------------------")

        # Initialize Lasso regression model
        lasso = Lasso(alpha=alpha)

        # Fit the model
        lasso.fit(x_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = lasso.predict(x_test_scaled)
        print(f'Predicted values: {y_pred[:5]}')
        print(f'Actual values: {y_test[:5]}')

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


@measure_time
def poly_reg(x_train, x_test, y_train, y_test):
    degrees = [1, 3, 4, ]

    for degree in degrees:
        print(f"Polynomial Regression Results for degree of {degree}: -------------------")

        poly_features = PolynomialFeatures(degree=degree)

        poly_regression = make_pipeline(poly_features, LinearRegression())

        y_pred_cv = cross_val_predict(poly_regression, x_train, y_train, cv=5)
        print(f'Predicted values: {y_pred_cv[:5]}')
        print(f'Actual values: {y_test[:5]}')

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


@measure_time
def decision_tree_regressor(x_train, x_test, y_train, y_test, viz=False):
    print("Decision Tree Regressor Results: -------------------")

    dt_reg = DecisionTreeRegressor(random_state=42)
    dt_reg.fit(x_train, y_train)

    y_pred_train = dt_reg.predict(x_train)
    y_pred_test = dt_reg.predict(x_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f'Train R-squared: {train_r2 * 100:.2f}%')
    print(f'Test R-squared: {test_r2 * 100:.2f}%')
    print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
    print(f'Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print(f'Test Mean Absolute Error (MAE): {test_mae:.2f}')

    if viz:
        plt.figure(figsize=(20, 10))
        tree.plot_tree(dt_reg, filled=True, feature_names=[f'Feature {i}' for i in range(len(x_train[0]))])
        plt.show()
        print()
        #
        # dot_file_path = 'Sol/results/decision_tree.dot'
        # export_graphviz(dt_reg, out_file=dot_file_path,
        #                 feature_names=[f'Feature {i}' for i in range(x_train.shape[1])],
        #                 filled=True, rounded=True, special_characters=True)
        #
        # # Read and visualize the dot file using graphviz
        # with open(dot_file_path) as f:
        #     dot_graph = f.read()
        # graphviz.Source(dot_graph).render("decision_tree", format="png", cleanup=True)
        # print(f"Decision tree saved as decision_tree.png")


@measure_time
def Hierach(x_train, x_test, y_train, y_test, truncate_mode=None, p=None, show_contracted=False,
            method='ward', metric='euclidean', criterion='maxclust', t=3,            viz=False):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler

    print("Hierarchical Clustering Results: -------------------")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(x_train)

    linked = linkage(scaled_data, method='ward')

    cluster_labels = fcluster(linked, t=t, criterion=criterion)

    # Evaluate clustering performance
    ari = adjusted_rand_score(y_train, cluster_labels)
    nmi = normalized_mutual_info_score(y_train, cluster_labels)
    fmi = fowlkes_mallows_score(y_train, cluster_labels)

    print(f'Adjusted Rand Index (ARI): {ari:.4f}')
    print(f'Normalized Mutual Information (NMI): {nmi:.4f}')
    print(f'Fowlkes-Mallows Index (FMI): {fmi:.4f}')
    print()

    if viz:
        plt.figure(figsize=(15, 10))

        dendrogram(
            linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True,
            truncate_mode=truncate_mode,
            p=30,
            show_contracted=show_contracted
        )

        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')
        plt.show()
        print()


@measure_time
def random_forest_regressor(x_train, x_test, y_train, y_test, viz=False):
    print("Random Forest Regressor Results: -------------------")

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(x_train, y_train)

    y_pred_train = rf_reg.predict(x_train)
    y_pred_test = rf_reg.predict(x_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f'Train R-squared: {train_r2 * 100:.2f}%')
    print(f'Test R-squared: {test_r2 * 100:.2f}%')
    print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
    print(f'Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print(f'Test Mean Absolute Error (MAE): {test_mae:.2f}')

    if viz:
        importances = rf_reg.feature_importances_
        feature_importances = sorted(zip(importances, range(len(importances))), reverse=True)

        print("Feature Importances:")
        for importance, feature_idx in feature_importances:
            print(f"Feature {feature_idx}: {importance:.4f}")

    print()


def kmeans_clustering(x_train, x_test):
    k_values = [2, 3, 4]

    for k in k_values:
        print(f"K-means Clustering Results for k = {k}: -------------------")

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x_train)

        y_pred_train = kmeans.predict(x_train)
        y_pred_test = kmeans.predict(x_test)

        silhouette_avg = silhouette_score(x_train, y_pred_train)

        print(f'Silhouette Score (Train): {silhouette_avg}')
        print(f'Cluster centers:\n{kmeans.cluster_centers_}')
        print()


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read_data()

    linear_regression(x_train, y_train, x_test, y_test)
    ridge_reg(x_train, x_test, y_train, y_test)
    lasso_reg(x_train, x_test, y_train, y_test)
    poly_reg(x_train, x_test, y_train, y_test)

    random_forest_regressor(x_train, x_test, y_train, y_test)

    decision_tree_regressor(x_train, x_test, y_train, y_test)

    Hierach(x_train, x_test, y_train, y_test)

    # aas = KNeighbors(x_train, x_test, y_train, y_test)
    # print(aas)
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
