import chess
from stockfish import Stockfish
from state import ChessState
import chess.engine

class PretrainedStockfishState(ChessState):
    _sf: Stockfish = None

    @classmethod
    def load_stockfish(cls, stockfish_path: str):
        cls._sf = Stockfish(
            path=stockfish_path,
            parameters={
                'Threads': 1,
                'Hash': 16
            }
        )
        cls._sf.set_skill_level(20)

    def __init__(self, board: chess.Board, parent=None, move = None):
        super().__init__(board, parent, move)

    def _compute_eval_score(self) -> float:
        if self._sf is None:
            self.load_stockfish(stockfish_path="./stockfish/stockfish-windows-x86-64-avx2.exe")

        if self.board.is_checkmate():
            return -99999.0 if self.board.turn == chess.WHITE else 99999.0
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0

        self._sf.set_fen_position(self.board.fen())
        ev = self._sf.get_evaluation()

        if ev["type"] == "cp":
            return float(max(-9000, min(9000, ev["value"])))

        return 99999.0 if ev["value"] > 0 else -99999.0


