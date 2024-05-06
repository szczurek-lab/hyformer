""" Base class for datasets."""


from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    """Base class for datasets."""

    def __init__(self) -> None:
        super().__init__()
        self.data = None
        self.target = None
        self.transform = None
        self.target_transform = None
        self._current = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        self._current += 1
        if self._current >= len(self.data):
            self._current = 0
            raise StopIteration
        else:
            idx = self._current - 1
            return self.__getitem__(idx)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target is None:
            return x
        else:
            y = self.target[idx]
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y

