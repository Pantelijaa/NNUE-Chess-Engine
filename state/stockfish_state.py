import os
import chess
import chess.engine
from typing import Optional

from state import ChessState


class StockfishState(ChessState):

    # Podrazumevano: release binarni + njegova originalna mreza.
    _stockfish_path: str = "./stockfish/stockfish-windows-x86-64-avx2.exe"
    _eval_file: Optional[str] = "./models/nn-c288c895ea92.nnue"
    _engine: Optional[chess.engine.SimpleEngine] = None

    def __init__(self, depth: int = 0):
        super().__init__(depth)

    @classmethod
    def _ensure_engine(cls) -> chess.engine.SimpleEngine:
        if cls.__dict__.get("_engine") is None:
            hash_mb = int(os.environ.get("STOCKFISH_HASH", "16"))
            engine = chess.engine.SimpleEngine.popen_uci(cls._stockfish_path)
            options = {"Threads": 1, "Hash": hash_mb, "Skill Level": 20}
            if cls._eval_file:
                options["EvalFile"] = cls._eval_file
            engine.configure(options)
            cls._engine = engine
        return cls._engine

    def _compute_eval_score(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -9999 + self.depth

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        engine = type(self)._ensure_engine()
        info = engine.analyse(board, chess.engine.Limit(depth=1))
        score = info["score"].white()
        mate = score.mate()
        if mate is not None:
            cp = 9000 if mate > 0 else -9000
        else:
            raw = score.score(mate_score=9000)
            cp = max(-9000, min(9000, raw if raw is not None else 0))

        return float(cp) if board.turn == chess.WHITE else -float(cp)

    @classmethod
    def _configured_classes(cls):
        result = [StockfishState]
        stack = list(StockfishState.__subclasses__())
        while stack:
            c = stack.pop()
            result.append(c)
            stack.extend(c.__subclasses__())
        return result

    @classmethod
    def close(cls):
        for klass in StockfishState._configured_classes():
            engine = klass.__dict__.get("_engine")
            if engine is not None:
                try:
                    engine.quit()
                except Exception:
                    pass
                klass._engine = None


def make_stockfish_state(stockfish_path: str,
                         eval_file: Optional[str] = None,
                         name: str = "StockfishStateConfigured") -> type:
    return type(name, (StockfishState,), {
        "_stockfish_path": stockfish_path,
        "_eval_file": eval_file,
        "_engine": None,
    })
