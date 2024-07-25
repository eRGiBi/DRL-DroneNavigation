import os
import itertools
import time
from functools import wraps

import numpy as np
import pandas as pd
import torch
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
from torch import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import optuna
import optuna.visualization as vis

import numpy as np

from torchviz import make_dot
import hiddenlayer as hl


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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=73)

    # x_train = np.array(x_train, dtype=np.float32)
    # print(x_train.max())
    # x_train = np.clip(x_train, -10, 10, out=x_train)
    # y_train = np.array(y_train, dtype=np.float32)
    # y_train = np.clip(y_train, -10, 10, out=y_train)
    # x_test = np.array(x_test, dtype=np.float32)
    # x_test = np.clip(x_test, -10, 10, out=x_test)
    # y_test = np.array(y_test, dtype=np.float32)
    # y_test = np.clip(y_test, -10, 10, out=y_test)

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
    degrees = [3, 4, 5]

    for degree in degrees:
        poly_plus(x_train, x_test, y_train, y_test, degree)


@measure_time
def poly_plus(x_train, x_test, y_train, y_test, degree):
    print(f"Polynomial Regression Results for degree of {degree}: -------------------")

    poly_features = PolynomialFeatures(degree=degree)

    poly_regression = make_pipeline(poly_features, LinearRegression())

    y_pred_cv = cross_val_predict(poly_regression, x_train, y_train, cv=5)
    y_pred_test = poly_regression.fit(x_train, y_train).predict(x_test)

    print(f'Predicted values (CV): {y_pred_cv[:5]}')
    print(f'Actual values: {y_test[:5]}')

    R_squared_train = r2_score(y_train, y_pred_cv)
    MSE_train = mean_squared_error(y_train, y_pred_cv)
    RMSE_train = mean_squared_error(y_train, y_pred_cv, squared=False)
    MAE_train = mean_absolute_error(y_train, y_pred_cv)
    MAPE_train = np.mean(np.abs((y_train - y_pred_cv) / y_train)) * 100  # Mean Absolute Percentage Error

    R_squared_test = r2_score(y_test, y_pred_test)

    print(f'For Degree of Polynomial: {degree}')
    print(f'Training R-squared: {round(R_squared_train * 100, 2)}%')
    print(f'Mean Absolute Error (MAE): {round(MAE_train, 2)}')
    print(f'Mean Squared Error (MSE): {round(MSE_train, 2)}')
    print(f'Root Mean Squared Error (RMSE): {round(RMSE_train, 2)}')
    print(f'Mean Absolute Percentage Error (MAPE): {round(MAPE_train, 2)}%')
    print(f'Test R-squared: {round(R_squared_test * 100, 2)}%')
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
            method='ward', metric='euclidean', criterion='maxclust', t=3, viz=False):
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


def optim_neural_net(x_train, x_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, pin_memory=True)

    class Net(nn.Module):
        def __init__(self, hidden_size1, hidden_size2, hidden_size3, hidden_size4):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(12, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.fc4 = nn.Linear(hidden_size3, hidden_size4)
            self.fc5 = nn.Linear(hidden_size4, 1)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = self.fc5(x)
            return x

    def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
        scaler = GradScaler()
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()

        return val_loss / len(val_loader)

    def objective(trial):
        hidden_size1 = trial.suggest_int('hidden_size1', 128, 640)
        hidden_size2 = trial.suggest_int('hidden_size2', 128, 640)
        hidden_size3 = trial.suggest_int('hidden_size3', 1, 512)
        hidden_size4 = trial.suggest_int('hidden_size4', 1, 360)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)

        model = Net(hidden_size1, hidden_size2, hidden_size3, hidden_size4).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               eps=1e-5)

        return train_model(model, criterion, optimizer, train_loader, val_loader)

    study = optuna.create_study(direction='minimize')

    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_model = Net(trial.params['hidden_size1'], trial.params['hidden_size2'],
                     trial.params['hidden_size3'], trial.params['hidden_size4'],
                     # trial.params['dropout_rate']
                     )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=trial.params['learning_rate'])

    for epoch in range(100):
        best_model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = best_model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

    best_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = best_model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            val_loss += loss.item()

    print("Final validation loss: ", val_loss / len(val_loader))

    vis.plot_param_importances(study).show()
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()


@measure_time
def used_neural_network(x_train, x_test, y_train, y_test):
    # x_train = np.array(x_train, dtype=np.float32)
    # print(x_train.max())
    # x_train = np.clip(x_train, -10, 10, out=x_train)
    # y_train = np.array(y_train, dtype=np.float32)
    # y_train = np.clip(y_train, -10, 10, out=y_train)
    # x_test = np.array(x_test, dtype=np.float32)
    # x_test = np.clip(x_test, -10, 10, out=x_test)
    # y_test = np.array(y_test, dtype=np.float32)
    # y_test = np.clip(y_test, -10, 10, out=y_test)

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    class Net(nn.Module):
        def __init__(self, ):
            super(Net, self).__init__()
            # self.extractor = nn.Flatten()
            self.fc1 = nn.Linear(12, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, 512)
            self.fc5 = nn.Linear(512, 256)
            self.fc6 = nn.Linear(256, 128)
            self.fc7 = nn.Linear(128, 64)
            self.fc8 = nn.Linear(64, 1)
            # self.fc6 = nn.Linear(256, 1)

        def forward(self, x):
            # x = self.extractor(x)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = torch.tanh(self.fc5(x))
            x = torch.tanh(self.fc6(x))
            x = torch.tanh(self.fc7(x))
            x = self.fc8(x)

            # x = self.fc6(x)
            # x = self.fc4(x)
            return x

    model = Net()
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 50
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Validation Loss: {val_loss / len(val_loader):.4f}"
        )

    model.eval()
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_train_pred = model(x_train_tensor).cpu().numpy()
        y_test_pred = model(x_test_tensor).cpu().numpy()

    with torch.no_grad():
        y_pred = model(x_test_tensor).cpu().numpy()

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f'Train R-squared: {train_r2 * 100:.2f}%')
    print(f'Test R-squared: {test_r2 * 100:.2f}%')
    print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
    print(f'Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print(f'Test Mean Absolute Error (MAE): {test_mae:.2f}')

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual', marker='o', linestyle='None')
    plt.plot(y_pred, label='Predicted', marker='x', linestyle='None')
    plt.legend()
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.show()

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

    sample_input = torch.randn(1, 12).to(device)

    # # Check if Graphviz is installed
    # if not os.system("dot -V"):
    #     # Visualize model using torchviz
    #     sample_input = torch.randn(1, 12).to(device)
    #     output = model(sample_input)
    #     dot = make_dot(output, params=dict(model.named_parameters()))
    #     dot.format = 'png'
    #     dot.render('model_architecture')
    # else:
    #     print("Graphviz not found. Skipping torchviz visualization.")
    #
    # # torch.onnx.export(model, sample_input, "model.onnx")
    #
    # # Visualize using hiddenlayer
    # # transforms = [hl.transforms.Prune('Constant')]
    # graph = hl.build_graph(model, sample_input)
    # graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save('model_hl.png')


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read_data()

    # linear_regression(x_train, y_train, x_test, y_test)
    # ridge_reg(x_train, x_test, y_train, y_test)
    # lasso_reg(x_train, x_test, y_train, y_test)
    # poly_reg(x_train, x_test, y_train, y_test)
    #
    # random_forest_regressor(x_train, x_test, y_train, y_test)
    #
    # decision_tree_regressor(x_train, x_test, y_train, y_test)

    # Hierach(x_train, x_test, y_train, y_test)
    #
    kmeans_clustering(x_train, x_test)

    # used_neural_network(x_train, x_test, y_train, y_test)
    #
    # optim_neural_net(x_train, x_test, y_train, y_test)

    aas = KNeighbors(x_train, x_test, y_train, y_test)
    print(aas)
    #
    # svm_acc = svm(x_train, y_train, x_test, y_test)
    #
    # print(f"Accuracy: {svm_acc}")

    # optim_svm = svm_param_search(x_train, y_train, x_test, y_test)
    #
    # print(f"Optimized SVM accuracy: {optim_svm}")
