import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score
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
    matches_df, unique_opponents = data_preprocessing(csv_path, 2018)

    # Creating tensors
    x_train_cat_tensor, x_train_num_tensor, y_train_tensor, x_test_cat_tensor, x_test_num_tensor, y_test_tensor = create_tensors(matches_df)

    # Creating Datasets
    train_dataset = MatchesDataset(x_train_cat_tensor, x_train_num_tensor, y_train_tensor)
    test_dataset = MatchesDataset(x_test_cat_tensor, x_test_num_tensor, y_test_tensor)

    # HIPERPARAMETERS
    EMBEDDING_VECTOR_SIZE = 10
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    EPOCHS = 300
    LR = 0.001

    # Creating model
    model = EkstraklasaMLP(len(unique_opponents), EMBEDDING_VECTOR_SIZE)

    # Creating criterion function: CrossEntropy for categorization and MSELoss for regression
    criterion_result = nn.CrossEntropyLoss()
    criterion_goals = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Parameters for DataLoaders
    shuffle_train = True
    shuffle_test = False

    # Creating DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=shuffle_train,
                                  num_workers=NUM_WORKERS)

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=shuffle_test,
                                  num_workers=NUM_WORKERS)

    # If GPU is available use GPU if not use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_per_epoch = []

    for epoch in range(EPOCHS):
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
            print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {loss_result.item()}")

    fig, ax = plt.subplots()
    ax.plot(loss_per_epoch)
    plt.show()

    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for x_cat_tensor, x_num_tensor, y_tensor in test_dataloader:

            x_cat_tensor = x_cat_tensor.to(device)
            x_num_tensor = x_num_tensor.to(device)
            y_tensor = y_tensor.to(device)

            result_labels_batch = y_tensor[:, 0].long()

            test_result = model(x_cat_tensor, x_num_tensor)

            result_loss = criterion_result(test_result, result_labels_batch)

            _, predicted_results_test = torch.max(test_result, 1)

            all_true_labels.append(result_labels_batch.cpu().numpy())
            all_predicted_labels.append(result_labels_batch.cpu().numpy())

    all_true_labels = np.concatenate(all_true_labels)
    all_predicted_labels = np.concatenate(all_predicted_labels)

    print(f"Loss on results test: {result_loss}")

    print("Confusion matrix:")
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    print(cm)

    target_names = ['Loss (0)', 'Draw (1)', 'Win (2)']
    print("\n Classification Report")
    print(classification_report(all_true_labels, all_predicted_labels, target_names=target_names))

    weighted_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    print(f"Weighted average F1-Score: {weighted_f1:.4f}")

if __name__ == "__main__":
    main()
