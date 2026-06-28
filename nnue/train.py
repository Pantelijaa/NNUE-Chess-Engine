from pandas.core.indexes import period
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
from . import DatasetStreamer, HalfKPNNUE


def train_nnue(model, dataset_path, epochs=3, batch_size=4096, lr=0.0001, print_freq=100):
    dataset = DatasetStreamer(dataset_path)
    train_loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        period_loss = 0.0
        batch_num = 0

        last_print_time = time.time()
        epoch_start_time = time.time()

        for batch_idx, (white_img, black_img, targets, stm) in enumerate(train_loader):
            white_img = white_img.to(device)
            black_img = black_img.to(device)
            targets = targets.to(device)
            stm = stm.to(device)

            optimizer.zero_grad()

            output = model(white_img, black_img, stm)
            loss = criterion(output, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()
            period_loss += loss.item()
            batch_num += 1

            if batch_num % print_freq == 0:
                current_time = time.time()
                elapsed_time = current_time - last_print_time

                speed = print_freq / elapsed_time
                avg_period_loss = period_loss / print_freq
                print(f"[Batch {batch_num}] -> Trenutni MSE Loss: {avg_period_loss:.6f} | Brzina: {speed:.1f} batch/s")

                period_loss = 0.0
                last_print_time = current_time

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        epoch_loss = running_loss / batch_num
        print(f"\n=> Epoha [{epoch + 1}/{epochs}] ZAVRŠENA!")
        print(f"   Ukupno procesirano batch-eva: {batch_num} (Ukupno pozicija: {batch_num * batch_size})")
        print(f"   Konačan prosečan MSE za celu epohu: {epoch_loss:.6f}")
        print(f"   Vreme trajanja epohe: {elapsed_time:.1f} sekundi")
        print(f"==================================================\n")