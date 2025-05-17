import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.models.EkstraklasaMLP import EkstraklasaMLP
from src.utils.MatchesDataset import MatchesDataset
from src.utils.create_tensors import create_tensors
from src.utils.data_preprocessing import data_preprocessing
import torch.nn as nn

# Loading and preprocessing data
csv_path = "data/poland_1.csv"
matches_df, unique_opponents = data_preprocessing(csv_path, 2010)

# Creating tensors
x_train_cat_tensor, x_train_num_tensor, y_train_tensor, x_test_cat_tensor, x_test_num_tensor, y_test_tensor = create_tensors(matches_df)

# Creating Datasets
train_dataset = MatchesDataset(x_train_cat_tensor, x_train_num_tensor, y_train_tensor)
test_dataset = MatchesDataset(x_test_cat_tensor, x_test_num_tensor, y_test_tensor)

# Creating model
model = EkstraklasaMLP(len(unique_opponents), 5, 3, 3, 1)

# Creating criterion function: CrossEntropy for categorization and MSELoss for regression
criterion_result = nn.CrossEntropyLoss()
criterion_goals = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Parameters for DataLoaders
batch_size = 32
shuffle_train = True
shuffle_test = False
num_workers = 2

# Creating DataLoaders
train_dataloader = DataLoader(train_dataset,
                              batch_size,
                              shuffle_train,
                              num_workers)

test_dataloader = DataLoader(test_dataset,
                              batch_size,
                              shuffle_test,
                              num_workers)

epochs = 20

for epoch in range(epochs):

    model.train()
