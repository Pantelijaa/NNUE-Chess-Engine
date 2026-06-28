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
        chess.KING: 9_000,
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

    def __init__(self, depth: int = 0):
        super().__init__(depth)

    def _compute_eval_score(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -9999 + self.depth
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        score += self._eval_material_value(board)
        score += self._eval_piece_value(board, chess.PAWN, self.PAWN_TABLE)
        score += self._eval_piece_value(board, chess.KNIGHT, self.KNIGHT_TABLE)
        score += self._eval_piece_value(board, chess.BISHOP, self.BISHOP_TABLE)
        score += self._eval_piece_value(board, chess.ROOK, self.ROOK_TABLE)
        score += self._eval_piece_value(board, chess.QUEEN, self.QUEEN_TABLE)

        mg_weight, eg_weight = self._get_game_phase_weights(board)
        king_mg_score = self._eval_piece_value(board, chess.KING, self.KING_MIDDLEGAME_TABLE)
        king_eg_score = self._eval_piece_value(board, chess.KING, self.KING_ENDGAME_TABLE)
        score += int((king_mg_score * mg_weight) + (king_eg_score * eg_weight))

        score += self._eval_passed_pawns(board)
        score += self._eval_pawn_structure(board)
        score += self._eval_bishop_pair(board)
        score += self._eval_rook_files(board)
        score += self._eval_mobility(board)
        score += self._eval_king_attacks(board)
        score += self._eval_rook_on_seventh(board)

        return float(score) if board.turn == chess.WHITE else -float(score)

    def _eval_material_value(self, board: chess.Board) -> int:
        """
            Calcuate total material based on pieces on board
        """
        score: int = 0
        for piece_type, value in self.PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, chess.WHITE))
            score -= value * len(board.pieces(piece_type, chess.BLACK))
        return score

    def _eval_piece_value(self, board: chess.Board, piece_type: chess.PieceType, piece_table: list) -> int:
        """
            Evaluate pieces value based on positions
        """
        score: int = 0
        for square in board.pieces(piece_type, chess.WHITE):
            score += piece_table[square]
        for square in board.pieces(piece_type, chess.BLACK):
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            score -= piece_table[rank * 8 + file]
        return score

    def _get_game_phase_weights(self, board: chess.Board) -> tuple[float, float]:
        """
        Game phase metric
        Returns (middlegame_weight, endgame_weight) scaled between 0.0 and 1.0
        """
        major_pieces = 0
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            major_pieces += len(board.pieces(pt, chess.WHITE))
            major_pieces += len(board.pieces(pt, chess.BLACK))

        mg_phase = min(14, major_pieces) / 14.0
        eg_phase = 1.0 - mg_phase
        return mg_phase, eg_phase

    def _eval_passed_pawns(self, board: chess.Board) -> int:
        score = 0

        for square in board.pieces(chess.PAWN, chess.WHITE):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            is_passed = True
            for rank in range(rank_idx + 1, 8):
                for file in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= file <= 7:
                        target_square = chess.square(file, rank)
                        if board.piece_at(target_square) == chess.Piece(chess.PAWN, chess.BLACK):
                            is_passed = False
                            break
                if not is_passed:
                    break

            if is_passed:
                score += 15 * rank_idx

        for square in board.pieces(chess.PAWN, chess.BLACK):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            is_passed = True
            for rank in range(0, rank_idx):
                for file in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= file <= 7:
                        target_square = chess.square(file, rank)
                        if board.piece_at(target_square) == chess.Piece(chess.PAWN, chess.WHITE):
                            is_passed = False
                            break
                if not is_passed:
                    break

            if is_passed:
                score -= 15 * (7 - rank_idx)

        return score

    def _eval_pawn_structure(self, board: chess.Board) -> int:
        """
        Prevent pawn structural weakness
        """
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            pawns = board.pieces(chess.PAWN, color)
            files = [chess.square_file(square) for square in pawns]
            for f in range(8):
                count = files.count(f)
                if count > 1:
                    score -= sign * 20 * (count - 1)
                if count >= 1:
                    if (f == 0 or (f - 1) not in files) and \
                            (f == 7 or (f + 1) not in files):
                        score -= sign * 15
        return score

    def _eval_bishop_pair(self, board: chess.Board) -> int:
        """
        Two bishops together are worth more than their individual value
        """
        score = 0
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += 50
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= 50
        return score

    def _eval_rook_files(self, board: chess.Board) -> int:
        """
        Prevent rooks being blocked by pawns
        """
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            for square in board.pieces(chess.ROOK, color):
                file = chess.square_file(square)
                file_mask = chess.BB_FILES[file]
                white_pawns = bool(board.pieces(chess.PAWN, chess.WHITE) & file_mask)
                black_pawns = bool(board.pieces(chess.PAWN, chess.BLACK) & file_mask)
                if not white_pawns and not black_pawns:
                    # open file
                    score += sign * 25
                elif not (white_pawns if color == chess.WHITE else black_pawns):
                    # semi-open file
                    score += sign * 15
        return score

    def _eval_mobility(self, board: chess.Board) -> int:
        """
        Sum each side attack squares
        Prefer active piece placement
        """
        white_mob = 0
        black_mob = 0
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for square in board.pieces(pt, chess.WHITE):
                white_mob += chess.popcount(board.attacks_mask(square))
            for square in board.pieces(pt, chess.BLACK):
                black_mob += chess.popcount(board.attacks_mask(square))
        result = (white_mob - black_mob) * 3
        return result

    def _eval_king_attacks(self, board: chess.Board) -> int:
        """
        Count attacks in 9-square range around king
        """
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            enemy_king = board.king(not color)
            if enemy_king is None:
                continue
            king_zone = chess.BB_KING_ATTACKS[enemy_king] | chess.BB_SQUARES[enemy_king]
            attacks_on_zone = 0
            for square in chess.scan_forward(king_zone):
                attacks_on_zone += chess.popcount(board.attackers_mask(color, square))
            score += sign * attacks_on_zone * 5
        return score

    def _eval_rook_on_seventh(self, board: chess.Board) -> int:
        """
        Bonus for rook on 7th rank (white) / 2nd rank (black)
        """
        score = 0
        for square in board.pieces(chess.ROOK, chess.WHITE):
            if chess.square_rank(square) == 6:
                score += 30
        for square in board.pieces(chess.ROOK, chess.BLACK):
            if chess.square_rank(square) == 1:
                score -= 30
        return score