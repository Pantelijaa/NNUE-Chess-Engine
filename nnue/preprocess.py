import chess
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from nnue import get_halfkp_indices
import pandas as pd
import os
import time
import numpy as np


def process_batch(df_dict):
    rows = []
    df = pd.DataFrame(df_dict)
    for _, row in df.iterrows():
        try:
            fen, cp, mate = row["fen"], row["cp"], row["mate"]
            board = chess.Board(fen)
            result = get_halfkp_indices(board)
            if result is None:
                continue
            w_idx, b_idx = result

            w_arr = np.zeros(41024, dtype=np.uint8)
            b_arr = np.zeros(41024, dtype=np.uint8)

            for idx in w_idx:
                if idx < 41024:
                    w_arr[idx] = 1
            for idx in b_idx:
                if idx < 41024:
                    b_arr[idx] = 1

            if mate is not None and not pd.isna(mate):
                cp_val = 30.0 * (0.99 ** (abs(int(mate)) - 1))
                if mate < 0:
                    cp_val = -cp_val
            else:
                cp_val = float(cp)
                cp_val = max(-3000.0, min(3000.0, cp_val)) / 100.0

            stm = 1 if board.turn == chess.WHITE else 0
            if stm == 0:
                cp_val = -cp_val
            cp_val = cp_val / 30.0

            rows.append({
                "w_features": w_arr.tobytes(),
                "b_features": b_arr.tobytes(),
                "cp": float(cp_val),
                "stm": int(stm)
            })
        except Exception:
            continue
    return rows

def process_file(parquet_path, output_dir):
    out_path = os.path.join(output_dir, "processed_" + os.path.basename(parquet_path))
    if os.path.exists(out_path):
        print(f"Preskačem (već postoji): {out_path}")
        return

    writer = None
    total = 0
    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows
    processed = 0
    start = time.time()
    last_print = time.time()

    with Pool(cpu_count()) as pool:
        for batch in parquet_file.iter_batches(batch_size=10000, columns=["fen", "cp", "mate"]):
            df = batch.to_pandas()
            sub_batches = [df[i:i + 650].to_dict('list') for i in range(0, len(df), 650)]

            results = pool.map(process_batch, sub_batches)
            all_rows = [row for result in results for row in result]
            total += len(all_rows)
            processed += len(df)

            now = time.time()
            elapsed = now - start
            speed = processed / elapsed
            eta = (num_rows - processed) / speed if speed > 0 else 0

            print(
                f"\r  {processed:,} / {num_rows:,} ({100 * processed / num_rows:.1f}%) | "
                f"{speed:,.0f} poz/s | "
                f"ETA: {eta:.0f}s",
                end="", flush=True
            )

            if all_rows:
                table = pa.Table.from_pylist(all_rows)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                writer.write_table(table)

            del df, all_rows, table, results, sub_batches

    if writer:
        writer.close()

    elapsed = time.time() - start
    print(f"\nObrađeno: {os.path.basename(parquet_path)} → {total:,} pozicija za {elapsed:.1f}s ({total/elapsed:,.0f} poz/s)")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder sa originalnim parquet fajlovima")
    parser.add_argument("--output", required=True, help="Folder za processed fajlove")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith('.parquet')
    ])
    print(f"Ukupno fajlova: {len(files)}")

    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {f}")
        process_file(f, args.output)

    print("\nPreprocessing završen!")