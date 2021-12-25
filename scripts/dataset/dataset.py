import pandas as pd
import torch
from torch.utils.data import Dataset

from scripts.data.windowing import get_windows
from scripts.dataset.utils import data_to_tensor


class WindowedDataset(Dataset):

    def __init__(self, x: pd.DataFrame, window: int):
        self.x = x
        self.pos_windows = None
        self.neg_windows = None
        self.norm_windows = None
        self._generate_window_data(x, window)

    def _generate_window_data(self, x: pd.DataFrame, window: int):
        self.norm_windows = get_windows(x, window, anomaly=0)
        self.pos_windows = get_windows(x, window, anomaly=1)
        self.neg_windows = get_windows(x, window, anomaly=-1)

        return

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data = data_to_tensor(self.norm_windows[index])

        return data

    def __len__(self):
        return len(self.norm_windows)

    def shape(self):
        return self.norm_windows[0].shape
