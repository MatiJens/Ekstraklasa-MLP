from torch.utils.data import Dataset


class MatchesDataset(Dataset):
    def __init__(self, cat_features, num_features, targets):
        self.cat_features = cat_features
        self.num_features = num_features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.cat_features[item], self.num_features[item], self.targets[item]