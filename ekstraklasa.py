import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.EkstraklasaMLP import EkstraklasaMLP
from src.utils.MatchesDataset import MatchesDataset
from src.utils.create_tensors import create_tensors
from src.utils.data_preprocessing import data_preprocessing
import torch.nn as nn

def main():
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
    optimizer = optim.Adam(model.parameters(), lr=0.005)

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

    epochs = 200
    loss_per_epoch = []

    for epoch in range(epochs):
        # Turning model in training mode
        model.train()

        for x_cat_tensor, x_num_tensor, y_tensor in train_dataloader:

            # Transfer tensor to GPU if available
            x_cat_tensor = x_cat_tensor.to(device)
            x_num_tensor = x_num_tensor.to(device)
            y_tensor = y_tensor.to(device)

            # Transform labels to correct format
            result_labels = y_tensor[:, 0].long()
            #goals_labels = y_tensor[:, 1].float().unsqueeze(1)

            # Zeroing gradients
            optimizer.zero_grad()

            # Model outputs
            predicted_results = model(x_cat_tensor, x_num_tensor)

            # Calculating loss
            loss_result = criterion_result(predicted_results, result_labels)
            #loss_goals = criterion_goals(predicted_goals, goals_labels)
            #total_loss = loss_goals + loss_result

            loss_result.backward()

            optimizer.step()

        loss_per_epoch.append(loss_result.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} Loss: {loss_result.item()}")

    fig, ax = plt.subplots()
    ax.plot(loss_per_epoch)
    plt.show()

    model.eval()

    with torch.no_grad():

        for x_cat_tensor, x_num_tensor, y_tensor in test_dataloader:

            x_cat_tensor = x_cat_tensor.to(device)
            x_num_tensor = x_num_tensor.to(device)
            y_tensor = y_tensor.to(device)

            result_labels = y_tensor[:, 0]
            #goals_labels = y_tensor[:, 1]

            test_result = model(x_cat_tensor, x_num_tensor)

            result_loss = criterion_result(test_result, result_labels)
            #goals_loss = criterion_goals(test_goals, goals_labels)

            _, predicted_results_test = torch.max(test_result, 1)
            correct_results = (predicted_results_test == result_labels).sum().item()

            total_prediction = result_labels.size(0)

            accuracy_results = correct_results / total_prediction

            print(f"Loss on results test: {result_loss}")
            #print(f"MSE Loss on results test: {goals_loss}")
            print(f"Results accuracy: {accuracy_results * 100:.2f}")


if __name__ == "__main__":
    main()
