import torch
import torch.nn as nn

class EkstraklasaMLP(nn.Module):
    def __init__(self, num_unique_team, embedding_vector_size):

        super(EkstraklasaMLP, self).__init__()

        # Creating two embedding layers: for away and home results
        self.home_embedding = nn.Embedding(num_unique_team, embedding_vector_size)
        self.away_embedding = nn.Embedding(num_unique_team, embedding_vector_size)

        # Calculating number of input neurons
        input_size = embedding_vector_size + embedding_vector_size + 3

        # Create mlp layers as Sequential
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Output layer
        self.fc_result = nn.Linear(16, 3)

    def forward(self, cat_inputs_batch, num_inputs_batch):

        # Split home and away inputs
        home_batch = cat_inputs_batch[:, 0]
        away_batch = cat_inputs_batch[:, 1]

        # Creating embedding vectors
        home_embedded = self.home_embedding(home_batch)
        away_embedded = self.away_embedding(away_batch)

        # Connecting embedding vectors and numerical features
        input_features = torch.cat((home_embedded, away_embedded, num_inputs_batch), dim=1)

        mlp_output = self.mlp(input_features)

        result_output = self.fc_result(mlp_output)

        return result_output


