import torch
import torch.nn as nn

class EkstraklasaMLP(nn.Module):
    def __init__(self, num_unique_team, embedding_vector_size, num_numerical_features, num_result_classes, output_goals_dim):

        super(EkstraklasaMLP, self).__init__()

        # Creating two embedding layers: for away and home results
        self.home_embedding = nn.Embedding(num_unique_team, embedding_vector_size)
        self.away_embedding = nn.Embedding(num_unique_team, embedding_vector_size)

        # Calculating number of input neurons
        input_size = embedding_vector_size + embedding_vector_size + num_numerical_features

        # Create mlp layers as Sequential
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Output layers
        self.fc_result = nn.Linear(16, num_result_classes)
        self.fc_goals = nn.Linear(16, output_goals_dim)

    def forward(self, cat_inputs_batch, num_inputs_batch):

        # Split home and away inputs and transforming them to correct form using squeeze
        home_batch = cat_inputs_batch[:, 0].squeeze()
        away_batch = cat_inputs_batch[:, 1].squeeze()

        # Creating embedding vectors
        home_embedded = self.home_embedding(home_batch)
        away_embedded = self.away_embedding(away_batch)

        # Connecting embedding vectors and numerical features
        input_features = torch.cat((home_embedded, away_embedded, num_inputs_batch), dim=1)

        mlp_output = self.mlp(input_features)

        result_output = self.fc_result(mlp_output)
        goals_output = self.fc_goals(mlp_output)

        return result_output, goals_output


