import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        return {key: torch.tensor(value[index]) for key, value in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
