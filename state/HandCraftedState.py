import chess
from state.ChessState import ChessState


class HandCraftedState(ChessState):
    """
        State klasa za Agenta 1 (PVS + rucna heuristika)
    """
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 10000,
    }

    PAWN_TABLE = [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ]

    KNIGHT_TABLE = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ]

    BISHOP_TABLE = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ]

    ROOK_TABLE = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ]

    QUEEN_TABLE = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ]

    KING_TABLE = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    ]

    def __init__(self, board: chess.Board, parent=None, move: chess.Move | None = None):
        super().__init__(board, parent, move)

    def get_eval_score(self) -> float:
        if self.board.is_checkmate():
            return -9999 if self.board.turn == chess.WHITE else 9999
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0

        score += self._eval_material_value()
        score += self._eval_piece_value(chess.PAWN, self.PAWN_TABLE)
        score += self._eval_piece_value(chess.KNIGHT, self.KING_TABLE)
        score += self._eval_piece_value(chess.BISHOP, self.BISHOP_TABLE)
        score += self._eval_piece_value(chess.ROOK, self.ROOK_TABLE)
        score += self._eval_piece_value(chess.QUEEN, self.QUEEN_TABLE)
        score += self._eval_piece_value(chess.KING, self.KING_TABLE)

        return score

    def _eval_material_value(self) -> int:
        score: int = 0
        for piece_type, value in self.PIECE_VALUES.items():
            score += value * len(self.board.pieces(piece_type, chess.WHITE))
            score -= value * len(self.board.pieces(piece_type, chess.BLACK))
        return score

    def _eval_piece_value(self, piece_type: chess.PieceType, piece_table: list) -> int:
        score: int = 0
        for square in self.board.pieces(piece_type, chess.WHITE):
            score += piece_table[square]
        for square in self.board.pieces(piece_type, chess.BLACK):
            score -= piece_table[63 - square]
        return score
