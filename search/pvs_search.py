import time
from typing import Optional
from collections import defaultdict
import chess
import chess.polyglot

from . import ChessSearch

TT_SCORE = 10_000 # Transposition table hit
PROMOTION_SCORE = 8_000 # Promocija pijuna
MVV_LVA_BASE = 5_000 # Most Valuable Victim - Least Valuable Attacker
KILLER_1_SCORE = 1_400 # Killer potez slot 1
KILLER_2_SCORE = 1_200 # Killer potez slot 2

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10_000,
}


class PVSSearch(ChessSearch):
    def __init__(self, time_limit:float= 1.0):
        super().__init__(time_limit)

        self._tt: dict = {}
        self._killer =  [[None, None] for _ in range(64)]
        self._history = defaultdict(int)

        # Dodatna statistika za PVS
        self.cutoffs = 0 # broj alpa-beta cutoffova
        self.tt_hits = 0 # broj transposition table hitova
        self.null_windows = 0 # broj null-window pretraga

    def best_move(self, board: chess.Board, state_class):
        self._reset_statistics()
        self._tt.clear()
        start = time.time()

        best_move = list(board.legal_moves)[0]
        depth = 1

        while True:
            elapsed = time.time() - start
            if elapsed >= self.time_limit:
                break
            try:
                score, move = self._root_search(board, state_class, depth, start)
                if move is not None:
                    best_move = move
                    self.depth_reached = depth
            except TimeoutError:
                break
            depth += 1

        self.time_limit = time.time() - start

        return best_move

    def _root_search(self, board, state_class, depth: int, start) -> tuple[float, chess.Move | None]:
        """Vraca (best_move, best_score)"""
        initial_state = state_class(board)
        alpha = -float('inf')
        beta = float('inf')

        best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
        score: float = -best_score
        best_move: chess.Move | None = None

        ordered = self._order_moves(board, depth, tt_move=self._get_tt_move(board))

        for move in ordered:
            if time.time() - start >= self.time_limit:
                raise TimeoutError
            board.push(move)
            try:
                state = state_class(board, parent=initial_state, move=move)

                if best_move is None:
                    score = self._pvs(state, depth - 1, alpha, beta, start) # full search za prvu granu
                else:
                    score = self._pvs(state, depth - 1, alpha, alpha + 1, start) # null window za ostale grane
                    self.null_windows += 1
                    if alpha < score < beta:
                        score = self._pvs(state, depth - 1, alpha, beta, start)  # re-search
            finally:
                board.pop()

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score, best_move = score, move
                    alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score, best_move = score, move
                    beta = min(beta, score)

        if best_move:
            self._store_tt(board, depth, best_score, best_move)

        return best_score, best_move


    def _pvs(self, state, depth: int, alpha: float, beta: float, start: float) -> float:
        """Rekurzivni PVS"""
        if time.time() - start >= self.time_limit:
            raise TimeoutError

        in_check = state.board.is_check()
        if in_check:
            depth += 1

        tt_entry = self._lookup_tt(state.board)
        if tt_entry and tt_entry["depth"] >= depth:
            self.tt_hits += 1
            return tt_entry["score"]

        if depth <= 0:
            return self._quiescence(state, alpha, beta, start)

        if state.is_final_state():
            return state.get_eval_score()

        tt_move = tt_entry["best_move"] if tt_entry else None
        ordered = self._order_moves(state.board, depth, tt_move=tt_move)
        first = True
        best_score = -float('inf') if state.board.turn == chess.WHITE else float('inf')
        best_move = None

        for move in ordered:
            next_state = state.__class__(state.board, parent=state, move=move)
            if first:
                score = self._pvs(next_state, depth - 1, alpha, beta, start)
                first = False
            else:
                score = self._pvs(next_state, depth - 1, alpha, alpha + 1, start) # null window
                self.null_windows += 1
                if alpha < score < beta:
                    score = self._pvs(next_state, depth - 1, alpha, beta, start) # Re-search sa punim prozorom

            maximizing = state.board.turn == chess.WHITE

            if maximizing:
                if score > best_score:
                    best_score, best_move = score, move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score, best_move = score, move
                beta = min(beta, score)

            if beta <= alpha:
                self.cutoffs += 1
                if not state.board.is_capture(move) and move.promotion is None:
                    self._update_killer(move, depth)
                    self._update_history(move, depth)
                break

        if best_move:
            self._store_tt(state.board, depth, best_score, best_move)

        return  best_score

    def _quiescence(self, state, alpha: float, beta: float, start: float) -> float:
        if time.time() - start >= self.time_limit:
            raise TimeoutError

        stand_pat = state.get_eval_score()
        maximizing = state.board.turn == chess.WHITE

        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)

            for move in state.board.generate_legal_moves(chess.BB_ALL, state.board.occupied_co[chess.BLACK]):
                next_state = state.__class__(state.board, parent=state, move=move)
                score = self._quiescence(next_state, alpha, beta, start)

                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            return alpha
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

            for move in state.board.generate_legal_moves(chess.BB_ALL, state.board.occupied_co[chess.WHITE]):
                next_state = state.__class__(state.board, parent=state, move=move)
                score = self._quiescence(next_state, alpha, beta, start)

                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            return beta

    # MOVE ORDERING

    def _order_moves(self, board: chess.Board, depth: int, tt_move: Optional[chess.Move]) -> list[chess.Move]:
        """Sortira poteze od najboljeg do najgoreg"""
        scored: list[tuple[int, chess.Move]] = []
        for move in board.legal_moves:
            score = self._score_move(board, move, depth, tt_move)
            scored.append((score, move))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [move for (score, move) in scored]

    def _score_move(self, board : chess.Board, move: chess.Move, depth: int, tt_move: Optional[chess.Move]) -> int:
        """Racuna score za rangiranje svakog poteza"""
        if move == tt_move:
            return TT_SCORE

        if move.promotion is not None:
            return PROMOTION_SCORE + PIECE_VALUES.get(move.promotion, 0)

        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                return (MVV_LVA_BASE
                        + PIECE_VALUES.get(victim.piece_type, 0)
                        - PIECE_VALUES.get(attacker.piece_type, 0) // 10
                )
            return MVV_LVA_BASE

        if depth < len(self._killer):
            if move == self._killer[depth][0]:
                return KILLER_1_SCORE
            if move == self._killer[depth][1]:
                return KILLER_2_SCORE

        moving_piece = board.piece_at(move.from_square)
        if moving_piece and moving_piece.piece_type == chess.PAWN:
            is_center_file = chess.square_file(move.to_square) in [2, 3, 4, 5]
            return 800 + (100 if is_center_file else 0)

        return min(self._history[move.uci()], 700)

    def _update_killer(self, move, depth: int):
        if depth < len(self._killer):
            if move != self._killer[depth][0]:
                self._killer[depth][1] = self._killer[depth][0]
                self._killer[depth][0] = move

    def _update_history(self, move: chess.Move, depth: int):
        self._history[move.uci()] += depth * depth

    # TT
    def _hash(self, board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board)

    def _lookup_tt(self, board: chess.Board) -> Optional[dict]:
        return  self._tt.get(self._hash(board))

    def _get_tt_move(self, board: chess.Board) -> Optional[chess.Move]:
        entry = self._lookup_tt(board)
        return entry["best_move"] if entry else None

    def _store_tt(self, board: chess.Board, depth: int, score: float, best_move: chess.Move):
        self._tt[self._hash(board)] = {
            "score": score,
            "depth": depth,
            "best_move": best_move
        }

    # STATISTICS
    def _reset_statistics(self):
        super()._reset_statistics()
        self.cutoffs = 0
        self.tt_hits = 0
        self.null_windows = 0

    def _get_statistics(self) -> dict:
        stats = super()._get_statistics()
        stats.update({
            "null_windows": self.null_windows,
            "tt_hits": self.tt_hits,
            "cutoffs": self.cutoffs,
            "cutoff_rate": round(self.cutoffs / max(1, self.nodes_visited), 3),
        })
        return stats