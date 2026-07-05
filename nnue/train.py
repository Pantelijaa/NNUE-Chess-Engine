from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import glob
import random
import numpy as np
from . import DatasetStreamer
from .dataset_streamer import load_val_sample

VAL_ROWS = 32768   # size of the fixed held-out validation batch

_last_saved_path = None

def _worker_init(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def save_best(model, epoch, batch_num, loss):
    global _last_saved_path
    os.makedirs("models", exist_ok=True)
    if _last_saved_path and os.path.exists(_last_saved_path):
        os.remove(_last_saved_path)
    path = f"models/nnue_e{epoch+1}_b{batch_num}_mse{loss:.6f}.pt"
    torch.save(model.state_dict(), path)
    _last_saved_path = path
    print(f"   >> Saved best model: {path}")


@torch.no_grad()
def _validate(model, val_batch, criterion):
    """Forward-only MSE on the fixed val batch; stable metric across the run."""
    model.eval()
    w_flat, w_off, b_flat, b_off, targets, stm = val_batch
    out = model(w_flat, w_off, b_flat, b_off, stm)
    mse = criterion(out, targets).item()
    mae = (out - targets).abs().mean().item()
    model.train()
    return mse, mae


def train_nnue(model, dataset_path, epochs=10, batch_size=8192, lr=1e-3, print_freq=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    model.train()

    files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards in {dataset_path}")
    val_path = files[-1] if len(files) > 1 else files[0]
    exclude = [val_path] if len(files) > 1 else []
    val_batch = tuple(t.to(device) for t in load_val_sample(val_path, VAL_ROWS))
    print(f"Validation: {VAL_ROWS} rows from {os.path.basename(val_path)} "
          f"({'held out' if exclude else 'overlaps train (single shard)'})")

    dataset = DatasetStreamer(dataset_path, batch_size=batch_size, exclude=exclude)
    train_loader = DataLoader(dataset, batch_size=None, num_workers=8, prefetch_factor=4,
                              pin_memory=True, persistent_workers=True, worker_init_fn=_worker_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=8,
        threshold=1e-4,
        cooldown=3,
        min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        running_loss = 0.0
        running_mae = 0.0
        period_loss = 0.0
        period_mae = 0.0
        batch_num = 0

        last_print_time = time.time()
        epoch_start_time = time.time()

        for batch_idx, (w_flat, w_offsets, b_flat, b_offsets, targets, stm) in enumerate(train_loader):
            w_flat = w_flat.to(device)
            w_offsets = w_offsets.to(device)
            b_flat = b_flat.to(device)
            b_offsets = b_offsets.to(device)
            targets = targets.to(device)
            stm = stm.to(device)

            optimizer.zero_grad()

            output = model(w_flat, w_offsets, b_flat, b_offsets, stm)
            loss = criterion(output, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            l = loss.item()
            batch_mae = (output.detach() - targets).abs().mean().item()
            running_loss += l
            running_mae += batch_mae
            period_loss += l
            period_mae += batch_mae
            batch_num += 1

            if batch_num % print_freq == 0:
                current_time = time.time()
                elapsed_time = current_time - last_print_time
                speed = print_freq / elapsed_time
                avg_period_loss = period_loss / print_freq
                avg_period_mae = period_mae / print_freq

                val_mse, val_mae = _validate(model, val_batch, criterion)
                scheduler.step(val_mse)
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"[Batch {batch_num}] -> train MSE: {avg_period_loss:.6f} MAE: {avg_period_mae:.6f} "
                      f"| val MSE: {val_mse:.6f} MAE: {val_mae:.6f} | Brzina: {speed:.1f} batch/s | LR: {cur_lr:.2e}")

                if val_mse < best_loss:
                    best_loss = val_mse
                    save_best(model, epoch, batch_num, best_loss)

                period_loss = 0.0
                period_mae = 0.0
                last_print_time = current_time

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        epoch_loss = running_loss / batch_num
        epoch_mae = running_mae / batch_num
        val_mse, val_mae = _validate(model, val_batch, criterion)
        print()
        print(f"=> Epoha [{epoch + 1}/{epochs}] ZAVRSENA!")
        print(f"   Ukupno procesirano batch-eva: {batch_num} (Ukupno pozicija: {batch_num * batch_size})")
        print(f"   Prosecan train MSE (epoha): {epoch_loss:.6f} | MAE: {epoch_mae:.6f}")
        print(f"   Val MSE: {val_mse:.6f} | MAE: {val_mae:.6f}")
        print(f"   Najbolji val MSE do sada: {best_loss:.6f}")
        print(f"   Vreme trajanja epohe: {elapsed_time:.1f} sekundi")
        print("==================================================")
        print()
