import tkinter as tk
from tkinter import messagebox
import chess
import threading
import time

from tournament import agent_map


class ChessTournamentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NNUE Chess Engine - Tournament Visualizer")
        self.root.geometry("600x650")
        self.root.configure(bg="#262421")

        # Initialize board state
        self.board = chess.Board()
        self.game_running = False

        # Colors for the chess board
        self.light_color = "#eeeed2"
        self.dark_color = "#769656"
        self.highlight_color = "#baca44"

        # Unicode characters for chess pieces
        self.piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
            '.': ''
        }

        self._setup_ui()
        self._draw_board()

    def _setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, bg="#312e2b", padx=10, pady=10)
        control_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(control_frame, text="White:", fg="white", bg="#312e2b", font=("Arial", 10, "bold")).grid(row=0,
                                                                                                          column=0,
                                                                                                          padx=5)
        self.white_menu = tk.StringVar(value="Handcrafted")
        tk.OptionMenu(control_frame, self.white_menu, *agent_map.keys()).grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="Black:", fg="white", bg="#312e2b", font=("Arial", 10, "bold")).grid(row=0,
                                                                                                          column=2,
                                                                                                          padx=5)
        self.black_menu = tk.StringVar(value="Handcrafted")
        tk.OptionMenu(control_frame, self.black_menu, *agent_map.keys()).grid(row=0, column=3, padx=5)

        self.start_btn = tk.Button(control_frame, text="Start Match", command=self.start_game, bg="#45a049", fg="white",
                                   font=("Arial", 10, "bold"))
        self.start_btn.grid(row=0, column=4, padx=15)

        # Status Label
        self.status_label = tk.Label(self.root, text="Select agents and press Start", fg="#b5b3b1", bg="#262421",
                                     font=("Arial", 12, "italic"))
        self.status_label.pack(pady=5)

        # Chessboard Container Grid
        self.board_frame = tk.Frame(self.root, bg="#262421")
        self.board_frame.pack(pady=10)

        self.squares = {}
        for r in range(8):
            for c in range(8):
                # We align rows to match chess standards (White on bottom, row 7 is index 0 visually)
                square_color = self.light_color if (r + c) % 2 == 0 else self.dark_color
                lbl = tk.Label(self.board_frame, text="", font=("Arial", 32), width=2, height=1, bg=square_color,
                               fg="black")
                lbl.grid(row=r, column=c)

                # Map coordinate space back to python-chess square integer system
                square_idx = chess.square(c, 7 - r)
                self.squares[square_idx] = lbl

    def _draw_board(self, last_move=None):
        for square_idx, label in self.squares.items():
            piece = self.board.piece_at(square_idx)
            symbol = self.piece_symbols[piece.symbol()] if piece else ""
            label.config(text=symbol)

            # Keep background accurate, accounting for move highlights
            row = 7 - chess.square_rank(square_idx)
            col = chess.square_file(square_idx)
            base_color = self.light_color if (row + col) % 2 == 0 else self.dark_color

            if last_move and (square_idx == last_move.from_square or square_idx == last_move.to_square):
                label.config(bg=self.highlight_color)
            else:
                label.config(bg=base_color)

    def start_game(self):
        if self.game_running:
            return

        self.board = chess.Board()
        self.game_running = True
        self.start_btn.config("disabled")

        # Fire engine processing onto a background worker thread to protect the UI thread loop
        threading.Thread(target=self._game_loop, daemon=True).start()

    def _game_loop(self):
        w_name = self.white_menu.get()
        b_name = self.black_menu.get()

        w_search, w_state = agent_map[w_name]
        b_search, b_state = agent_map[b_name]

        # Use 1-second limits for snappy visual play
        white_agent = w_search(time_limit=1.0)
        black_agent = b_search(time_limit=1.0)

        while not self.board.is_game_over() and self.board.fullmove_number <= 150:
            turn_start = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"{turn_start} calculation loop ongoing...")

            if self.board.turn == chess.WHITE:
                move = white_agent.best_move(self.board, w_state)
            else:
                move = black_agent.best_move(self.board, b_state)

            # Safely play move onto visualization board tracking instance
            self.board.push(move)

            # Thread-safe updates dispatched back down to main Tkinter UI handler
            self.root.after(0, self._draw_board, move)
            time.sleep(0.1)  # brief structural delay to capture placement changes

        # Wrap up processing
        res = self.board.result()
        outcome = "Draw!" if res == "1/2-1/2" else "White Wins!" if res == "1-0" else "Black Wins!"

        self.status_label.config(text=f"Game Over: {outcome}")
        self.start_btn.config("normal")
        self.game_running = False
        messagebox.showinfo("Tournament Match Concluded", f"Final Result: {res}\n({outcome})")