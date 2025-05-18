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
    matches_df_train, matches_df_test, unique_opponents = data_preprocessing(csv_path, 2018, 2023)

    # Creating tensors
    x_train_cat_tensor, x_train_num_tensor, y_train_tensor, x_test_cat_tensor, x_test_num_tensor, y_test_tensor = create_tensors(matches_df_train, matches_df_test)

    # Creating Datasets
    train_dataset = MatchesDataset(x_train_cat_tensor, x_train_num_tensor, y_train_tensor)
    test_dataset = MatchesDataset(x_test_cat_tensor, x_test_num_tensor, y_test_tensor)

    # HIPERPARAMETERS
    EMBEDDING_VECTOR_SIZE = 10
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    EPOCHS = 100
    LR = 0.001

    # Creating model
    model = EkstraklasaMLP(len(unique_opponents), EMBEDDING_VECTOR_SIZE)

    # Creating criterion function: CrossEntropy for categorization and MSELoss for regression
    criterion_result = nn.CrossEntropyLoss()
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

        for x_train_cat_batch, x_train_num_batch, y_train_batch in train_dataloader:

            # Transfer tensor to GPU if available
            x_train_cat_batch = x_train_cat_batch.to(device)
            x_train_num_batch = x_train_num_batch.to(device)
            y_train_batch = y_train_batch.to(device)

            # Transform labels to correct format
            train_result_labels = y_train_batch[:, 0].long()

            # Zeroing gradients
            optimizer.zero_grad()

            # Model outputs
            train_predicted_results = model(x_train_cat_batch, x_train_num_batch)

            # Calculating loss
            loss_result_train = criterion_result(train_predicted_results, train_result_labels)

            loss_result_train.backward()

            optimizer.step()

        loss_per_epoch.append(loss_result_train.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {loss_result_train.item()}")

    fig, ax = plt.subplots()
    ax.plot(loss_per_epoch)
    plt.show()

    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for x_test_cat_batch, x_test_num_batch, y_test_batch in test_dataloader:

            x_test_cat_batch = x_test_cat_batch.to(device)
            x_test_num_batch = x_test_num_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            test_result_labels = y_test_batch[:, 0].long()

            test_result = model(x_test_cat_batch, x_test_num_batch)

            result_loss_test = criterion_result(test_result, test_result_labels)

            _, predicted_results_test = torch.max(test_result, 1)

            all_true_labels.append(test_result_labels.cpu().numpy())
            all_predicted_labels.append(predicted_results_test.cpu().numpy())

    all_true_labels = np.concatenate(all_true_labels)
    all_predicted_labels = np.concatenate(all_predicted_labels)

    print(f"Loss on results test: {result_loss_test.item()}")

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
