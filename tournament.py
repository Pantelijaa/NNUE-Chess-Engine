import chess

from search import PVSSearch, MCTSSearch
from state import HandCraftedState

agent_map = {
    "Handcrafted": (PVSSearch , HandCraftedState),
    "MCTS": (MCTSSearch, None)
}

class Tournament:
    def play_game(self, white_name: str, black_name: str):
        white_search_class, white_state_class = agent_map[white_name]
        black_search_class, black_state_class = agent_map[black_name]

        white_agent = white_search_class()
        black_agent = black_search_class()

        board = chess.Board()

        while not board.is_game_over() and board.fullmove_number <= 100:
            if board.turn == chess.WHITE:
                move = white_agent.best_move(board, white_state_class)
            else:
                move = black_agent.best_move(board, black_state_class)
            board.push(move)
            print(board.fen())

        result = board.result()
        return "white" if result == "1-0" else "black" if result == "0-1" else "draw"