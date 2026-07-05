import win_startup  # noqa: F401  # Windows/torch startup fix — mora pre torch importa

import tkinter as tk
from ui import ChessTournamentGUI
import argparse
from nnue.halfkp_nnue import HalfKPNNUE
from nnue.train import train_nnue

def main_gui():
    """
    Displays GUI with option to choose which agents will face each other.
    Used mainly for debugging purposes.
    :return:
    """
    root = tk.Tk()
    app = ChessTournamentGUI(root)
    root.mainloop()

def main_train():
    """
    NNNUE training pipeline.
    :return:
    """
    nnue_model = HalfKPNNUE()
    dataset_path = "./nnue/processed"
    train_nnue(nnue_model, dataset_path)

def main_tournament(time_limit: float, games_per_side: int, workers: int,
                    sf_threads: int, sf_hash: int):
    """
    Tournament simulation. No GUI. Progress is tracked in terminal and final results are saved locally.
    :param time_limit: Agent's maximum time budget per move
    :param games_per_side: How many games will each pair play before changing sides
    :param workers: How many games will be played at the same time. Cant exceed total number of CPU cores. If workers=1 No multiprocessing
    :param sf_threads: How many threads will be dedicated to Stockfish Evaluator. workers + sf_threads shouldn't exceed total number of CPU cores.
    :param sf_hash: Memory allocate for Stockfish Transposition Table. In MB.
    :return:
    """
    from tournament import Tournament
    t = Tournament(time_limit=time_limit, games_per_side=games_per_side)
    if workers > 1:
        t.run_parallel(workers=workers, sf_threads=sf_threads, sf_hash=sf_hash, verbose=True)
    else:
        t.run(verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-main script runner.")
    parser.add_argument("--mode", type=str, required=True, help="Which main function to run")
    parser.add_argument("--time-limit", type=float, default=1.0,
                        help="Vremenski budzet po potezu u sekundama (tournament mode)")
    parser.add_argument("--games-per-side", type=int, default=10,
                        help="Broj partija po strani za svaki par (tournament mode)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Broj paralelnih procesa (>1 ukljucuje paralelni turnir)")
    parser.add_argument("--sf-threads", type=int, default=1,
                        help="Broj niti po Stockfish procesu u paralelnom rezimu")
    parser.add_argument("--sf-hash", type=int, default=16,
                        help="Hash (MB) po Stockfish procesu u paralelnom rezimu")

    args = parser.parse_args()

    if args.mode == "gui":
        main_gui()
    elif args.mode == "train":
        main_train()
    elif args.mode == "tournament":
        main_tournament(args.time_limit, args.games_per_side, args.workers,
                        args.sf_threads, args.sf_hash)
    else:
        print(f"Unknown mode: {args.mode}")
