import torch
import torch.nn as nn

class EkstraklasaMLP(nn.Module):
    def __init__(self, num_unique_opponents, embedding_dim, num_numerical_features, hidden_layer_sizes, num_classes):
        super(EkstraklasaMLP, self).__init__()

        self.opponent_embedding = nn.Embedding(num_embeddings=num_unique_opponents, embedding_dim=embedding_dim)

        total_input_size = embedding_dim + num_numerical_features

        layers = []
        input_size = total_input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
            layers.append(nn.Linear(input_size, num_classes))
            self.mlp = nn.Sequential(*layers)

    def forward(self, cat_features, num_features):

        opponent_id = cat_features[:, 0]

        opponent_embedding_vector = self.opponent_embedding(opponent_id)

        combined_features = torch.cat(
            (opponent_embedding_vector, num_features),
            dim = 1
        )

        output = self.mlp(combined_features)

        return output
