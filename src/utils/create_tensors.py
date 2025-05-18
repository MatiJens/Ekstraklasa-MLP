from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_tensors(data):
    # Definition of categorical and numerical feature columns and target column
    categorical_features = ['home', 'away']
    numerical_features = ['matchday', 'last_results_home', 'last_results_away']
    target_column = ['result']

    # Creating train and test sets
    x = data[categorical_features + numerical_features]
    y = data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y['result'])

    scaler = StandardScaler()

    x_train_num_scaled = scaler.fit_transform(x_train[numerical_features])
    x_test_num_scaled = scaler.transform(x_test[numerical_features])

    x_train_cat_tensor = torch.LongTensor(x_train[categorical_features].values)
    x_train_num_tensor = torch.FloatTensor(x_train_num_scaled)
    y_train_tensor = torch.LongTensor(y_train.values)

    x_test_cat_tensor = torch.LongTensor(x_test[categorical_features].values)
    x_test_num_tensor = torch.FloatTensor(x_test_num_scaled)
    y_test_tensor = torch.LongTensor(y_test.values)

    return x_train_cat_tensor, x_train_num_tensor, y_train_tensor, x_test_cat_tensor, x_test_num_tensor, y_test_tensor