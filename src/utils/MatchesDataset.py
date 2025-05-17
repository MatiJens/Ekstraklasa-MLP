from torch.utils.data import Dataset

class MatchesDataset(Dataset):
    def __init__(self, x_cat_tensor, x_num_tensor, y_tensor):
        self.x_cat_tensor = x_cat_tensor
        self.x_num_tensor = x_num_tensor
        self.y_tensor = y_tensor

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.x_cat_tensor[idx], self.x_num_tensor[idx], self.y_tensor[idx]