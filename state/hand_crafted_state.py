import chess
from state.chess_state import ChessState


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
        chess.KING: 10_000,
    }

    PAWN_TABLE = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -25, -25, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 25, 25, 0, 0, 0,
        5, 5, 10, 27, 27, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
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

    KING_MIDDLEGAME_TABLE = [
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30
    ]

    KING_ENDGAME_TABLE = [
        -50, -30, -30, -30, -30, -30, -30, -50,
        -30, -10, -10, 0, 0, -10, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    ]

    def __init__(self, board: chess.Board, parent=None, move: chess.Move | None = None):
        super().__init__(board, parent, move)

    def _compute_eval_score(self) -> float:
        if self.board.is_checkmate():
            return -9999 + self.depth
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        score = 0
        score += self._eval_material_value()
        score += self._eval_piece_value(chess.PAWN, self.PAWN_TABLE)
        score += self._eval_piece_value(chess.KNIGHT, self.KNIGHT_TABLE)
        score += self._eval_piece_value(chess.BISHOP, self.BISHOP_TABLE)
        score += self._eval_piece_value(chess.ROOK, self.ROOK_TABLE)
        score += self._eval_piece_value(chess.QUEEN, self.QUEEN_TABLE)

        mg_weight, eg_weight = self._get_game_phase_weights()
        king_mg_score = self._eval_piece_value(chess.KING, self.KING_MIDDLEGAME_TABLE)
        king_eg_score = self._eval_piece_value(chess.KING, self.KING_ENDGAME_TABLE)
        score += int((king_mg_score * mg_weight) + (king_eg_score * eg_weight))

        score += self._eval_passed_pawns()

        if self.board.turn == chess.WHITE:
            return float(score)
        else:
            return float(-score)

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
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            score -= piece_table[rank * 8 + file]
        return score

    def _get_game_phase_weights(self) -> tuple[float, float]:
        """Returns (middlegame_weight, endgame_weight) scaled between 0.0 and 1.0"""
        major_pieces = 0
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            major_pieces += len(self.board.pieces(pt, chess.WHITE))
            major_pieces += len(self.board.pieces(pt, chess.BLACK))

        mg_phase = min(14, major_pieces) / 14.0
        eg_phase = 1.0 - mg_phase
        return mg_phase, eg_phase

    def _eval_passed_pawns(self) -> int:
        score = 0

        for square in self.board.pieces(chess.PAWN, chess.WHITE):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            is_passed = True
            for rank in range(rank_idx + 1, 8):
                for file in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= file <= 7:
                        target_square = chess.square(file, rank)
                        if self.board.piece_at(target_square) == chess.Piece(chess.PAWN, chess.BLACK):
                            is_passed = False
                            break
                if not is_passed:
                    break

            if is_passed:
                score += 15 * rank_idx

        for square in self.board.pieces(chess.PAWN, chess.BLACK):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            is_passed = True
            for rank in range(0, rank_idx):
                for file in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= file <= 7:
                        target_square = chess.square(file, rank)
                        if self.board.piece_at(target_square) == chess.Piece(chess.PAWN, chess.WHITE):
                            is_passed = False
                            break
                if not is_passed:
                    break

            if is_passed:
                score -= 15 * (7 - rank_idx)

        return score
