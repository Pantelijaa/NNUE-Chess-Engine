import chess
import numpy as np

NUM_FEATURES = 41024

def get_piece_offset(piece: chess.Piece, is_black_perspective: bool) -> int:
    p_color = piece.color
    if is_black_perspective:
        p_color = not p_color
    color_offset = 0 if p_color == chess.WHITE else 5
    return (color_offset + (piece.piece_type - 1)) * 64


def get_halfkp_indices(board: chess.Board):
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return None
    bk_f = chess.square_mirror(bk)

    w_idx = []
    b_idx = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        w_idx.append(wk * 512 + get_piece_offset(piece, False) + sq)
        b_idx.append(bk_f * 512 + get_piece_offset(piece, True) + chess.square_mirror(sq))
    return w_idx, b_idx


class NNUEInference:
    """
    All network weights as numpy arrays + a fast forward pass.
    The result is the side-to-move-relative score normalised to ~[-1, 1].
    """

    def __init__(self):
        self.ft_weight = None  # [41024, 256]
        self.ft_bias = None    # [256]
        self.w1 = None         # [512, 32]
        self.b1 = None         # [32]
        self.w2 = None         # [32, 32]
        self.b2 = None         # [32]
        self.w3 = None         # [32, 1]
        self.b3 = None         # [1]

    @classmethod
    def from_model(cls, model):
        self = cls()
        self.ft_weight = model.ft_weight.detach().numpy().astype(np.float32)  # [41024, 256]
        self.ft_bias = model.ft_bias.detach().numpy().astype(np.float32)      # [256]

        sd = model.hidden_layers.state_dict()
        self.w1 = sd["0.weight"].numpy().T.astype(np.float32).copy()  # [512, 32]
        self.b1 = sd["0.bias"].numpy().astype(np.float32)            # [32]
        self.w2 = sd["2.weight"].numpy().T.astype(np.float32).copy()  # [32, 32]
        self.b2 = sd["2.bias"].numpy().astype(np.float32)            # [32]
        self.w3 = sd["4.weight"].numpy().T.astype(np.float32).copy()  # [32, 1]
        self.b3 = sd["4.bias"].numpy().astype(np.float32)            # [1]
        return self

    @staticmethod
    def _scrl(x):
        # squared clipped relu; clip returns a fresh array so callers' inputs are safe
        c = np.clip(x, 0.0, 1.0)
        c *= c
        return c

    def evaluate(self, stm_vec, opp_vec) -> float:
        d = stm_vec.shape[0]                 # transformer width
        a_stm = self._scrl(stm_vec)
        a_opp = self._scrl(opp_vec)

        # row 0 = [stm, opp], row 1 = [opp, stm]  -> one matmul yields both terms
        x = np.empty((2, d * 2), dtype=np.float32)
        x[0, :d] = a_stm
        x[0, d:] = a_opp
        x[1, :d] = a_opp
        x[1, d:] = a_stm

        h = self._scrl(x @ self.w1 + self.b1)
        h = self._scrl(h @ self.w2 + self.b2)
        out = h @ self.w3 + self.b3   # [2, 1]
        return float(out[0, 0] - out[1, 0])


class Accumulator:
    """
    Holds the two perspective accumulators (pre-activation) for one position.
    Ideally should be incrementally updated but in python that is actually slower
    """

    def __init__(self, net: NNUEInference):
        self.net = net
        self.white = net.ft_bias.copy()
        self.black = net.ft_bias.copy()

    def refresh(self, board: chess.Board):
        net = self.net
        res = get_halfkp_indices(board)
        if res is None:
            self.white = net.ft_bias.copy()
            self.black = net.ft_bias.copy()
            return
        w_idx, b_idx = res
        W = net.ft_weight
        self.white = net.ft_bias + W[w_idx].sum(axis=0)
        self.black = net.ft_bias + W[b_idx].sum(axis=0)
