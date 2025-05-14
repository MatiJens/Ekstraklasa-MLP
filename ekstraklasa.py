import torch
from torch import optim
from torch.utils.data import DataLoader

from src.models.EkstraklasaMLP import EkstraklasaMLP
from src.utils.MatchesDataset import MatchesDataset
from src.utils.create_tensors import create_tensors
from src.utils.load_data_from_csv import load_data_from_csv
import torch.nn as nn

csv_path = "data/poland_1.csv"

matches_df = load_data_from_csv(csv_path, "Legia Warszawa", 2005)

X_train_cat_tensor, X_train_num_tensor, y_train_tensor, X_test_cat_tensor, X_test_num_tensor, y_test_tensor = create_tensors(matches_df)

train_dataset = MatchesDataset(X_train_cat_tensor, X_train_num_tensor, y_train_tensor)
test_dataset = MatchesDataset(X_test_cat_tensor, X_test_num_tensor, y_test_tensor)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_cat_ids = torch.cat([X_train_cat_tensor, X_test_cat_tensor])
num_unique_cat_values = len(torch.unique(all_cat_ids))

num_numerical_features = X_train_num_tensor.shape[1]

num_classes = len(torch.unique(y_train_tensor))

EMBEDDING_DIM = 8
HIDDEN_LAYER_SIZES = [64, 32]

model = EkstraklasaMLP(
    num_unique_cat_values,
    EMBEDDING_DIM,
    num_numerical_features,
    HIDDEN_LAYER_SIZES,
    num_classes
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.001)

EPOCHS = 20

for epoch in range(epochs):

    model.train()
