from abc import ABC, abstractmethod
import chess
import chess.polyglot


class ChessState(ABC):
    _eval_cache: dict[int, float] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._eval_cache = {}

    @abstractmethod
    def __init__(self, board, parent=None, move=None):
        self.parent = parent
        self.move = move
        if parent is None:
            self.board = board.copy()
            self.depth: int = 0
        else:
            self.board = parent.board.copy()
            self.board.push(self.move)
            self.depth = parent.depth + 1

    def get_eval_score(self) -> float:
        if self.board.is_game_over():
            return self._compute_eval_score()
        key = chess.polyglot.zobrist_hash(self.board)
        cached = type(self)._eval_cache.get(key)
        if cached is not None:
            return cached
        result = self._compute_eval_score()
        type(self)._eval_cache[key] = result
        return result

    @abstractmethod
    def _compute_eval_score(self) -> float:
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

    @classmethod
    def clear_eval_cache(cls):
        cls._eval_cache.clear()