import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from multiprocessing import Pool, cpu_count
from nnue import get_halfkp_indices
import pandas as pd
import os


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
                "w_indices": w_idx,
                "b_indices": b_idx,
                "cp": cp_val,
                "stm": stm
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

    with Pool(cpu_count()) as pool:
        for batch in parquet_file.iter_batches(batch_size=40000, columns=["fen", "cp", "mate"]):
            df = batch.to_pandas()
            sub_batches = [df[i:i + 2000].to_dict('list') for i in range(0, len(df), 2000)]

            results = pool.map(process_batch, sub_batches)
            all_rows = [row for result in results for row in result]
            total += len(all_rows)
    
            if all_rows:
                table = pa.Table.from_pylist(all_rows)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                writer.write_table(table)

    if writer:
        writer.close()
    print(f"Obrađeno: {parquet_path} → {total} pozicija")


if __name__ == "__main__":
    input_dir = "./dataset"
    output_dir = "./processed/"
    os.makedirs(output_dir, exist_ok=True)

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet")]
    print(f"Ukupno fajlova: {len(files)}")

    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {f}")
        process_file(f, output_dir)

    print("Preprocessing završen!")