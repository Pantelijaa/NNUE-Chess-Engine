from abc import ABC, abstractmethod
import chess
import chess.polyglot


class ChessState(ABC):
    _eval_cache: dict[int, float] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._eval_cache = {}

    @abstractmethod
    def __init__(self, depth: int = 0):
        self.depth = depth

    def make_child(self) -> "ChessState":
        return type(self)(self.depth)

    def get_eval_score(self, board: chess.Board) -> float:
        if board.is_game_over():
            return self._compute_eval_score(board)
        key = chess.polyglot.zobrist_hash(board)
        cached = type(self)._eval_cache.get(key)
        if cached is not None:
            return cached
        result = self._compute_eval_score(board)
        type(self)._eval_cache[key] = result
        return result

    @abstractmethod
    def _compute_eval_score(self, board: chess.Board ) -> float:
        pass

    def is_final_state(self, board: chess.Board) -> bool:
        return board.is_game_over()

    @classmethod
    def clear_eval_cache(cls):
        cls._eval_cache.clear()