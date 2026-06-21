import chess
import torch.nn as nn
import torch
import numpy as np

def get_piece_offset(piece: chess.Piece, is_black_perspective: bool) -> int:
    p_type = piece.piece_type
    p_color = piece.color

    if is_black_perspective:
        p_color = not p_color

    color_offset = 0 if p_color == chess.WHITE else 5
    type_idx = p_type - 1
    return (color_offset + type_idx) * 64

def get_halfkp_indices(board: chess.Board):
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    if white_king_sq is None or black_king_sq is None:
        return None
    white_king_sq_flipped = chess.square_mirror(white_king_sq)
    black_king_sq_flipped = chess.square_mirror(black_king_sq)

    white_indices = []
    black_indices = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue

        pt_offset_white = get_piece_offset(piece, is_black_perspective=False)
        white_idx = white_king_sq * 512 + pt_offset_white + square
        white_indices.append(white_idx)

        square_flipped = chess.square_mirror(square)
        pt_offset_black = get_piece_offset(piece, is_black_perspective=True)
        black_idx = black_king_sq_flipped * 512 + pt_offset_black + square_flipped
        black_indices.append(black_idx)

    return white_indices, black_indices

class Accumulator():
    def __init__(self, transformer_layer: nn.Linear):
        self.weights = transformer_layer.weight.detach().numpy().T # [41024, 256]
        self.bias = transformer_layer.bias.detach().numpy() # [256]

        self.white_state = np.zeros(256, dtype=np.float32)
        self.black_state = np.zeros(256, dtype=np.float32)

    def refresh_from_scratch(self, board: chess.Board):
        white_indices, black_indices = get_halfkp_indices(board)

        self.white_state = np.copy(self.bias)
        self.black_state = np.copy(self.bias)

        for idx in white_indices:
            if idx < 41024:
                self.white_state += self.weights[idx]
        for idx in black_indices:
            if idx < 41024:
                self.black_state += self.weights[idx]

    def update_move(self, removed_white, added_white, removed_black, added_black):
        for idx in removed_white:
            if idx < 41024:
                self.white_state -= self.weights[idx]
        for idx in added_white:
            if idx < 41024:
                self.white_state += self.weights[idx]

        for idx in removed_black:
            if idx < 41024:
                self.black_state -= self.weights[idx]
        for idx in added_black:
            if idx < 41024:
                self.black_state += self.weights[idx]

    def to_tensor(self):
        w_tensor = torch.from_numpy(self.white_state).unsqueeze(0)
        b_tensor = torch.from_numpy(self.black_state).unsqueeze(0)

        return w_tensor, b_tensor