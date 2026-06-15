from abc import ABC, abstractmethod
import chess


class ChessState(ABC):
    @abstractmethod
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.parent: ChessState  = parent
        self.move: chess.Move = move
        if parent is None:
            self.board: chess.Board = board.copy()
            self.depth: int = 0
        else:
            self.board = parent.board.copy()
            self.board.push(self.move)
            self.depth = parent.depth + 1

    @abstractmethod
    def get_eval_score(self) -> float:
        pass

    def get_next_states(self) -> list:
        return [self.__class__(self.board, parent=self, move=self.move) for move in self.board.legal_moves]

    def get_unique_hash(self) -> str:
        return self.board.fen()

    def is_final_state(self) -> bool:
        return self.board.is_game_over()

    def reconstruct_path(self) -> list[chess.Move]:
        path = []
        state = self
        while state.parent is not None:
            path.append(state.move)
            state = state.parent
        return list(reversed(path))
