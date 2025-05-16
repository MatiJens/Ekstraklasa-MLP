import torch
from torch import optim
from torch.utils.data import DataLoader

from src.models.EkstraklasaMLP import EkstraklasaMLP
from src.utils.MatchesDataset import MatchesDataset
from src.utils.create_tensors import create_tensors
from src.utils.load_data_from_csv import load_data_from_csv
import torch.nn as nn

csv_path = "data/poland_1.csv"

matches_df, unique_opponents = load_data_from_csv(csv_path, 2024)

print(matches_df.info())
print(unique_opponents)

#X_train_cat_tensor, X_train_num_tensor, y_train_tensor, X_test_cat_tensor, X_test_num_tensor, y_test_tensor = create_tensors(matches_df)

#train_dataset = MatchesDataset(X_train_cat_tensor, X_train_num_tensor, y_train_tensor)
#test_dataset = MatchesDataset(X_test_cat_tensor, X_test_num_tensor, y_test_tensor)
