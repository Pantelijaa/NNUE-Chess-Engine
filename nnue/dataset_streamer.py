import os
import random
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import IterableDataset
import pyarrow as pa
import pyarrow.parquet as pq

_COLS = ["w_indices", "b_indices", "cp", "stm"]


def _embedding_input(col):
    """T
    urn an Arrow list<int> column into EmbeddingBag (flat_values, offsets).
    Arrow already stores a list column as a flat child buffer + offsets, which isexactly EmbeddingBag's input format
    """
    arr = col.combine_chunks()                       # ChunkedArray -> ListArray
    offsets = arr.offsets.to_numpy()                 # int32, length n+1
    start, end = int(offsets[0]), int(offsets[-1])   # slicing can leave start != 0
    flat = arr.values.to_numpy(zero_copy_only=False)[start:end]
    # .astype(copy) -> writable, contiguous arrays (Arrow buffers are read-only)
    flat_t = torch.from_numpy(flat.astype(np.int64))
    off_t = torch.from_numpy((offsets[:-1] - start).astype(np.int64))
    return flat_t, off_t


def table_to_tensors(table):
    w_flat, w_off = _embedding_input(table.column("w_indices"))
    b_flat, b_off = _embedding_input(table.column("b_indices"))
    cp = torch.from_numpy(table.column("cp").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32)).unsqueeze(1)
    stm = torch.from_numpy(table.column("stm").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32))
    return w_flat, w_off, b_flat, b_off, cp, stm

def load_val_sample(parquet_path, num_rows, num_groups=16, seed=0):
    """Representative fixed val set: sample rows from row-groups spread across the
    whole shard (not the first contiguous rows, which are a narrow biased slice)."""
    pf = pq.ParquetFile(parquet_path)
    ng = pf.num_row_groups
    rng = np.random.default_rng(seed)
    groups = sorted(set(np.linspace(0, ng - 1, min(num_groups, ng)).astype(int).tolist()))
    per = num_rows // len(groups) + 1
    parts = []
    for g in groups:
        table = pf.read_row_group(g, columns=_COLS)
        idx = np.sort(rng.permutation(table.num_rows)[:per])
        parts.append(table.take(idx))
    return table_to_tensors(pa.concat_tables(parts).slice(0, num_rows))


class DatasetStreamer(IterableDataset):
    def __init__(self, folder_path, batch_size=8192, shuffle=True,
                 buffer_size=None, read_chunk=None, exclude=None):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size or batch_size * 32
        self.read_chunk = read_chunk or batch_size * 4
        self.exclude = set(exclude or ())

    def _files(self):
        files = sorted(
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith(".parquet")
        )
        if self.exclude:
            files = [f for f in files if f not in self.exclude]
        # shard FIRST (sorted list is identical across workers -> disjoint split),
        # THEN shuffle each worker's own slice. shuffling before sharding overlaps
        # because each worker has its own RNG.
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            files = files[worker.id::worker.num_workers]
        if self.shuffle:
            random.shuffle(files)
        return files

    def _to_tensors(self, table):
        return table_to_tensors(table)

    def _emit_full_batches(self, table):
        if self.shuffle:
            table = table.take(np.random.permutation(table.num_rows))
        n = table.num_rows
        full = (n // self.batch_size) * self.batch_size
        for start in range(0, full, self.batch_size):
            yield self._to_tensors(table.slice(start, self.batch_size))
        return table.slice(full)

    def __iter__(self):
        pending = None
        parts = []            # accumulated row-group tables (the shuffle buffer)
        rows = 0

        for path in self._files():
            pf = pq.ParquetFile(path)
            groups = list(range(pf.num_row_groups))
            if self.shuffle:
                random.shuffle(groups)
            for g in groups:
                t = pf.read_row_group(g, columns=_COLS)
                parts.append(t)
                rows += t.num_rows
                if rows >= self.buffer_size:
                    tables = ([pending] if pending is not None else []) + parts
                    pending = yield from self._emit_full_batches(pa.concat_tables(tables))
                    parts, rows = [], 0

        # flush whatever is left
        tables = ([pending] if pending is not None else []) + parts
        if tables:
            leftover = yield from self._emit_full_batches(pa.concat_tables(tables))
            if leftover.num_rows > 0:
                yield self._to_tensors(leftover)
