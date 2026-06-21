import tkinter as tk
from ui import ChessTournamentGUI
import argparse
from nnue.halfkp_nnue import HalfKPNNUE
from nnue.train import train_nnue

def main_gui():
    root = tk.Tk()
    app = ChessTournamentGUI(root)
    root.mainloop()

def main_train():
    nnue_model = HalfKPNNUE()
    dataset_path = "./nnue/processed"
    train_nnue(nnue_model, dataset_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-main script runner.")
    parser.add_argument("--mode", type=str, required=True, help="Which main function to run")

    args = parser.parse_args()

    if args.mode == "gui":
        main_gui()
    elif args.mode == "train":
        main_train()
    else:
        print(f"Unknown mode: {args.mode}")