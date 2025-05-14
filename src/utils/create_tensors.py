from sklearn.model_selection import train_test_split
import torch
import pandas as pd

def create_tensors(matches_df):

    # Definition of categorical and numerical feature columns and target column
    categorical_features = ['opponent_id']
    numerical_features = ['matchday', 'is_home', 'goals', 'last_results']
    target_column = 'result'

    # Creating train and test sets
    X = matches_df[categorical_features + numerical_features]
    y = matches_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_cat_tensor = torch.LongTensor(X_train[categorical_features].values)
    X_train_num_tensor = torch.FloatTensor(X_train[numerical_features].values)
    y_train_tensor = torch.LongTensor(y_train.values)

    X_test_cat_tensor = torch.LongTensor(X_test[categorical_features].values)
    X_test_num_tensor = torch.FloatTensor(X_test[numerical_features].values)
    y_test_tensor = torch.LongTensor(y_test.values)

    return X_train_cat_tensor, X_train_num_tensor, y_train_tensor, X_test_cat_tensor, X_test_num_tensor, y_test_tensor