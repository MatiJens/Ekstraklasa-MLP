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
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle_train,
                              num_workers=num_workers)

test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle_test,
                              num_workers=num_workers)

# If GPU is available use GPU if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 50

for epoch in range(epochs):
    # Turning model in training mode
    model.train()

    running_loss = 0.0
    correct_predictions = 0

    for x_cat_tensor, x_num_tensor, y_tensor in train_dataloader:

        # Transfer tensor to GPU if available
        x_cat_tensor = x_cat_tensor.to(device)
        x_num_tensor = x_num_tensor.to(device)
        y_tensor = y_tensor.to(device)

        # Transform labels to correct format
        result_labels = y_tensor[:, 0]#.long()
        goals_labels = y_tensor[:, 1]#.float().unsqueeze(1)

        # Zeroing gradients
        optimizer.zero_grad()

        # Model outputs
        predicted_results, predicted_goals = model(x_cat_tensor, x_num_tensor)

        # Calculating loss
        loss_result = criterion_result(predicted_results, result_labels)
        loss_goals = criterion_goals(predicted_goals, goals_labels)
        total_loss = loss_goals + loss_result

        total_loss.backward()

        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss.item()}")
