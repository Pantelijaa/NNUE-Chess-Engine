import chess
import chess.engine
from typing import Optional

from state import ChessState


class PretrainedStockfishState(ChessState):
    _engine: Optional[chess.engine.SimpleEngine] = None

    @classmethod
    def load_stockfish(cls, stockfish_path: str):
        cls._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        cls._engine.configure({"Threads": 1, "Hash": 16, "Skill Level": 20, "EvalFile": "./models/nn-c288c895ea92.nnue"})

    def __init__(self, depth: int = 0):
        super().__init__(depth)

    def _compute_eval_score(self, board: chess.Board) -> float:
        if self._engine is None:
            self.load_stockfish("./stockfish/stockfish-windows-x86-64-avx2.exe")

        if board.is_checkmate():
            return -9999 + self.depth

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        info = self._engine.analyse(board, chess.engine.Limit(depth=1))  # type: ignore[union-attr]
        score = info["score"].white()
        mate = score.mate()
        if mate is not None:
            cp = 9000 if mate > 0 else -9000
        else:
            raw = score.score(mate_score=9000)
            cp = max(-9000, min(9000, raw if raw is not None else 0))

        return float(cp) if board.turn == chess.WHITE else -float(cp)

    @classmethod
    def close(cls):
        if cls._engine is not None:
            cls._engine.quit()
            cls._engine = None
