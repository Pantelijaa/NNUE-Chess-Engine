import time
import chess
import chess.engine
from typing import Optional

from search.chess_search import ChessSearch

STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2.exe"


class StockfishSearch(ChessSearch):
    _engine: Optional[chess.engine.SimpleEngine] = None

    @classmethod
    def load_stockfish(cls):
        cls._engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        cls._engine.configure({"Threads": 1, "Hash": 16, "EvalFile": "./models/nn-c288c895ea92.nnue"})

    @classmethod
    def close(cls):
        if cls._engine is not None:
            cls._engine.quit()
            cls._engine = None

    def best_move(self, board: chess.Board, state_class=None) -> chess.Move:
        if self._engine is None:
            self.load_stockfish()

        self._reset_statistics()
        start = time.time()

        result = self._engine.play(
            board,
            chess.engine.Limit(time=self.time_limit),
            info=chess.engine.INFO_ALL,
        )

        self.time_elapsed = time.time() - start
        self.nodes_visited = result.info.get("nodes", 0)
        self.depth_reached = result.info.get("depth", 0)

        return result.move or list(board.legal_moves)[0]
