import torch.nn as nn

class EkstraklasaMLP(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):

        super(EkstraklasaMLP, self).__init__()

        embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)

    #def forward(self, x_cat, x_num):

