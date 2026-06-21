from cmath import nan

import chess
from torch.utils.data import IterableDataset
import pyarrow.dataset as ds
from .accumulator import get_halfkp_indices
import torch
import math


def indices_to_tensor(indices_list, num_features=41024):
    tensor = torch.zeros(num_features, dtype=torch.float32)
    valid_indices = [idx for idx in indices_list if idx < num_features]
    tensor[valid_indices] = 1.0
    return tensor

class DatasetStreamer(IterableDataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def __iter__(self):
        dataset = ds.dataset(self.folder_path, format='parquet')
        for batch in dataset.to_batches(columns=["w_indices", "b_indices", "cp", "stm"]):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                w_tensor = indices_to_tensor(row["w_indices"])
                b_tensor = indices_to_tensor(row["b_indices"])
                cp_target = torch.tensor([row["cp"] * 30 * 100], dtype=torch.float32)
                stm = torch.tensor(row["stm"])
                yield w_tensor, b_tensor, cp_target, stm

