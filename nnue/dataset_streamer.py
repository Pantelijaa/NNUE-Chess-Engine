from cmath import nan

import chess
from torch.utils.data import IterableDataset
import pyarrow.dataset as ds
from .accumulator import get_halfkp_indices
import torch
import math
import pyarrow.parquet as pq
import os
import numpy as np


def indices_to_tensor(indices_list, num_features=41024):
    tensor = torch.zeros(num_features, dtype=torch.float32)
    valid_indices = [idx for idx in indices_list if idx < num_features]
    tensor[valid_indices] = 1.0
    return tensor

class DatasetStreamer(IterableDataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def __iter__(self):
        files = sorted([
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith('.parquet')
        ])

        for file in files:
            pf = pq.ParquetFile(file)
            for batch in pf.iter_batches(batch_size=4096, columns=["w_features", "b_features", "cp", "stm"]):
                w = np.frombuffer(
                    b"".join(batch.column("w_features").to_pylist()),
                    dtype=np.uint8
                ).reshape(-1, 41024).copy()

                b = np.frombuffer(
                    b"".join(batch.column("b_features").to_pylist()),
                    dtype=np.uint8
                ).reshape(-1, 41024).copy()

                yield (
                    torch.from_numpy(w).float(),
                    torch.from_numpy(b).float(),
                    torch.tensor(batch.column("cp").to_pylist(), dtype=torch.float32).unsqueeze(1),
                    torch.tensor(batch.column("stm").to_pylist(), dtype=torch.int32)
                )

