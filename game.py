import chess
from game_result import GameResult


class Game:
    """
    Upravlja jednom partijom izmedju dva agenta.
    Analogno RobotGame.do_search() iz robot projekta.
    """

    MAX_MOVES = 200  # maksimalan broj polupoteza pre prisilnog remija

    def play(
        self,
        white_name:   str,
        black_name:   str,
        white_search,
        black_search,
        white_state,
        black_state,
        move_callback=None,
        verbose: bool = False,
    ) -> GameResult:
        """
        Odigrava jednu partiju izmedju dva agenta.

        Args:
            white_name:     ime belog agenta (za statistiku)
            black_name:     ime crnog agenta (za statistiku)
            white_search:   instanca ChessSearch za belog
            black_search:   instanca ChessSearch za crnog
            white_state:    state klasa za belog  (HandCraftedState, ...)
            black_state:    state klasa za crnog
            move_callback:  opcioni callback(board, move) — poziva se
                            posle svakog poteza (koristi GUI za crtanje)
            verbose:        stampa svaki potez u konzoli

        Returns:
            GameResult sa svim statistikama partije
        """
        board       = chess.Board()
        white_stats = []
        black_stats = []
        move_count  = 0

        while not board.is_game_over() and move_count < self.MAX_MOVES:

            if board.turn == chess.WHITE:
                move   = white_search.best_move(board, white_state)
                stats  = white_search.get_statistics()
                white_stats.append(stats)
            else:
                move   = black_search.best_move(board, black_state)
                stats  = black_search.get_statistics()
                black_stats.append(stats)

            board.push(move)
            move_count += 1

            # Obavesti GUI ili bilo kojeg posmatraca o novom potezu
            if move_callback:
                move_callback(board, move)

            if verbose:
                side = "Beli" if not board.turn else "Crni"
                print(
                    f"  {move_count:>3}. {side:5} igra {move} "
                    f"| nodes={stats['nodes_visited']:>7,} "
                    f"| depth={stats['depth_reached']:>2} "
                    f"| time={stats['time_elapsed']:.3f}s"
                )

        result      = board.result()
        termination = self._termination_reason(board, move_count)

        game_result = GameResult(
            white_name  = white_name,
            black_name  = black_name,
            result      = result,
            num_moves   = move_count,
            termination = termination,
            white_stats = white_stats,
            black_stats = black_stats,
        )
        game_result.compute_averages()

        if verbose:
            print(f"\n  Kraj: {result} ({termination}, {move_count} poteza)\n")

        return game_result

    def _termination_reason(self, board: chess.Board, move_count: int) -> str:
        if move_count >= self.MAX_MOVES:            return "move_limit"
        if board.is_checkmate():                    return "checkmate"
        if board.is_stalemate():                    return "stalemate"
        if board.is_insufficient_material():        return "insufficient_material"
        if board.is_seventyfive_moves():            return "75_moves"
        if board.is_fivefold_repetition():          return "fivefold_repetition"
        return "unknown"
