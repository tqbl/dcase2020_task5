import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MappedDataLoader(DataLoader):
    def __init__(self, dataset, device=None, **args):
        super().__init__(dataset, **args)
        self.device = device

    def __iter__(self):
        def _to(data):
            if isinstance(data, (tuple, list)):
                return tuple(_to(item) for item in data)
            return data.to(self.device)

        return map(_to, super().__iter__())


class SimpleDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = torch.FloatTensor(x).permute(0, 3, 1, 2)
        self.y = _tensor_or_none(y)
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)

        if self.y is None:
            return x,
        return x, self.y[index]

    def __len__(self):
        return len(self.x)


class ContextualizedDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = torch.FloatTensor(x[0]).permute(0, 3, 1, 2)
        self.aux = torch.FloatTensor(x[1])
        self.y = _tensor_or_none(y)
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        aux = self.aux[index]

        if self.y is None:
            return (x, aux),
        return (x, aux), self.y[index]

    def __len__(self):
        return len(self.x)


def _tensor_or_none(x):
    return torch.FloatTensor(x) if x is not None else None
