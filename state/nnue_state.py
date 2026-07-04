import os
import re
import chess
import torch

from state.chess_state import ChessState
from nnue import HalfKPNNUE
from nnue.accumulator import Accumulator, NNUEInference

# normalized network output -> centipawns:  cp = norm * 30 pawns * 100
CP_SCALE = 30.0 * 100.0

class NNUEState(ChessState):
    _model: HalfKPNNUE = None
    _net: NNUEInference = None
    _acc: Accumulator = None  # shared, refreshed per eval

    def __init__(self, depth: int = 0):
        if type(self)._net is None:
            self.load_model(models_dir="models")
        super().__init__(depth)

    def _compute_eval_score(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -9999 + self.depth
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        cls = type(self)
        acc = cls._acc
        acc.refresh(board)

        if board.turn == chess.WHITE:
            norm = cls._net.evaluate(acc.white, acc.black)
        else:
            norm = cls._net.evaluate(acc.black, acc.white)

        return norm * CP_SCALE  # side-to-move relative centipawns

    @classmethod
    def load_model(cls, models_dir: str = "models"):
        pattern = re.compile(r"mse([0-9]+\.[0-9]+)\.pt$")
        best_path, best_mse = None, float("inf")
        for fname in os.listdir(models_dir):
            m = pattern.search(fname)
            if m:
                mse = float(m.group(1))
                if mse < best_mse:
                    best_mse = mse
                    best_path = os.path.join(models_dir, fname)
        if best_path is None:
            raise FileNotFoundError(f"No NNUE model found in '{models_dir}'")
        print(f"Loading NNUE model: {best_path} (MSE {best_mse})")

        model = HalfKPNNUE()
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
        model.eval()
        cls._model = model
        cls._net = NNUEInference.from_model(model)
        cls._acc = Accumulator(cls._net)
