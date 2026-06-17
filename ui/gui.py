import tkinter as tk
from tkinter import messagebox
import chess
import threading
import time

from tournament import Tournament, agent_map


class ChessTournamentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NNUE Chess Engine - Tournament Visualizer")
        self.root.geometry("600x650")
        self.root.configure(bg="#262421")

        self.board = chess.Board()
        self.game_running = False
        self.tournament = Tournament(time_limit=1.0)

        self.light_color = "#eeeed2"
        self.dark_color = "#769656"
        self.highlight_color = "#baca44"

        self.piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
        }

        self._setup_ui()
        self._draw_board()

    def _setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#312e2b", padx=10, pady=10)
        control_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(control_frame, text="White:", fg="white", bg="#312e2b",
                 font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
        self.white_menu = tk.StringVar(value=list(agent_map.keys())[0])
        tk.OptionMenu(control_frame, self.white_menu,
                      *agent_map.keys()).grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="Black:", fg="white", bg="#312e2b",
                 font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5)
        self.black_menu = tk.StringVar(value=list(agent_map.keys())[0])
        tk.OptionMenu(control_frame, self.black_menu,
                      *agent_map.keys()).grid(row=0, column=3, padx=5)

        self.start_btn = tk.Button(
            control_frame, text="Start Match",
            command=self.start_game,
            bg="#45a049", fg="white", font=("Arial", 10, "bold")
        )
        self.start_btn.grid(row=0, column=4, padx=15)

        self.status_label = tk.Label(
            self.root, text="Select agents and press Start",
            fg="#b5b3b1", bg="#262421", font=("Arial", 12, "italic")
        )
        self.status_label.pack(pady=5)

        self.stats_label = tk.Label(
            self.root, text="",
            fg="#b5b3b1", bg="#262421", font=("Arial", 9)
        )
        self.stats_label.pack(pady=2)

        self.board_frame = tk.Frame(self.root, bg="#262421")
        self.board_frame.pack(pady=10)

        self.squares = {}
        for r in range(8):
            for c in range(8):
                color = self.light_color if (r + c) % 2 == 0 else self.dark_color
                lbl = tk.Label(
                    self.board_frame, text="",
                    font=("Arial", 32), width=2, height=1,
                    bg=color, fg="black"
                )
                lbl.grid(row=r, column=c)
                square_idx = chess.square(c, 7 - r)
                self.squares[square_idx] = lbl

    def _draw_board(self, board: chess.Board = None, last_move: chess.Move = None):
        if board is None:
            board = self.board

        for square_idx, label in self.squares.items():
            piece = board.piece_at(square_idx)
            symbol = self.piece_symbols.get(piece.symbol(), "") if piece else ""
            label.config(text=symbol)

            row = 7 - chess.square_rank(square_idx)
            col = chess.square_file(square_idx)
            color = self.light_color if (row + col) % 2 == 0 else self.dark_color

            if last_move and square_idx in (last_move.from_square, last_move.to_square):
                label.config(bg=self.highlight_color)
            else:
                label.config(bg=color)

        self.board = board


    def start_game(self):
        if self.game_running:
            return
        self.board = chess.Board()
        self.game_running = True
        self.start_btn.config(state="disabled")
        self._draw_board()
        threading.Thread(target=self._game_loop, daemon=True).start()

    def _game_loop(self):
        white_name = self.white_menu.get()
        black_name = self.black_menu.get()

        self._update_status(f"Partija: {white_name} vs {black_name}")

        result = self.tournament.play_game(
            white_name=white_name,
            black_name=black_name,
            move_callback=self._on_move,  # GUI callback
        )

        outcome = (
            "Remi!" if result.result == "1/2-1/2"
            else "Beli pobedio!" if result.result == "1-0"
            else "Crni pobedio!"
        )

        stats_text = (
            f"Potezi: {result.num_moves} | "
            f"{white_name} avg nodes: {result.white_avg_nodes:,.0f} | "
            f"{black_name} avg nodes: {result.black_avg_nodes:,.0f}"
        )

        self.root.after(0, self.stats_label.config, {"text": stats_text})
        self.root.after(0, self._update_status, f"Kraj: {outcome} ({result.termination})")
        self.root.after(0, self.start_btn.config, {"state": "normal"})
        self.root.after(0, messagebox.showinfo,
                        "Kraj partije",
                        f"Rezultat: {result.result}\n{outcome}\n\n{stats_text}")

        self.game_running = False

    def _on_move(self, board: chess.Board, move: chess.Move):
        side = "Crni" if board.turn == chess.WHITE else "Beli"
        self.root.after(0, self._update_status, f"{side} je odigrao {move}")
        self.root.after(0, self._draw_board, board.copy(), move)
        time.sleep(0.3)

    def _update_status(self, text: str):
        self.status_label.config(text=text)