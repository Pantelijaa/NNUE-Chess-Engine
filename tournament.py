import itertools
import json
from datetime import datetime
from typing import Callable, Optional

import chess

from search import PVSSearch, MCTSSearch
from state import HandCraftedState, PretrainedStockfishState
from game import Game
from game_result import GameResult

# ── agent_map ────────────────────────────────────────────────────────────────
# GUI citа ovaj recnik da popuni dropdown menije.
# Kljuc  = ime agenta prikazano u GUI
# Vrednost = (SearchClass, StateClass)
#
# Dodaj nove agente ovde kada budu implementirani:
#   from state import MiniNNUEState, MyStockfishState, PretrainedStockfishState
#   from search import MCTSSearch
#   "Mini NNUE":          (PVSSearch, MiniNNUEState),
#   "My Stockfish NNUE":  (PVSSearch, MyStockfishState),
#   "Pretrained SF NNUE": (PVSSearch, PretrainedStockfishState),
#   "MCTS":               (MCTSSearch, None),

agent_map = {
    "Handcrafted": (PVSSearch, HandCraftedState),
    "MCTS": (MCTSSearch, None),
    "PretrainedStockfish": (PVSSearch, PretrainedStockfishState)
}

class Tournament:
    def __init__(self, time_limit: float = 1.0, games_per_pair: int = 50):
        """
        Args:
            time_limit:     vremenski budzet po potezu u sekundama
            games_per_pair: broj partija po paru u automatskom turniru
                            (svaka strana odigra po games_per_pair kao beli)
        """
        self.time_limit = time_limit
        self.games_per_pair = games_per_pair
        self._game = Game()
        self._results: list[GameResult] = []

    def play_game(
            self,
            white_name: str,
            black_name: str,
            move_callback: Optional[Callable] = None,
            verbose: bool = False,
    ) -> GameResult:
        """
        Odigrava jednu partiju izmedju dva agenta iz agent_map.
        move_callback(board, move) se poziva posle svakog poteza — GUI
        koristi ovo da azurira prikaz table u realnom vremenu.

        Args:
            white_name:    kljuc iz agent_map za belog
            black_name:    kljuc iz agent_map za crnog
            move_callback: opcioni callback za GUI
            verbose:       stampa poteze u konzoli

        Returns:
            GameResult sa statistikama partije
        """
        white_search_cls, white_state_cls = agent_map[white_name]
        black_search_cls, black_state_cls = agent_map[black_name]

        white_search = white_search_cls(time_limit=self.time_limit)
        black_search = black_search_cls(time_limit=self.time_limit)

        result = self._game.play(
            white_name=white_name,
            black_name=black_name,
            white_search=white_search,
            black_search=black_search,
            white_state=white_state_cls,
            black_state=black_state_cls,
            move_callback=move_callback,
            verbose=verbose,
        )

        self._results.append(result)
        return result

    def run(self, verbose: bool = False) -> dict:
        names = list(agent_map.keys())
        pairs = list(itertools.combinations(names, 2))
        total = len(pairs) * self.games_per_pair * 2

        print(
            f"Turnir pocinje: {len(names)} agenata, "
            f"{len(pairs)} parova, {total} partija ukupno\n"
        )

        wins = {name: 0 for name in names}
        losses = {name: 0 for name in names}
        draws = {name: 0 for name in names}

        for name_a, name_b in pairs:
            print(f"  {name_a}  vs  {name_b}")
            a_wins = b_wins = pair_draws = 0

            for game_num in range(self.games_per_pair * 2):
                if game_num % 2 == 0:
                    white, black = name_a, name_b
                else:
                    white, black = name_b, name_a

                result = self.play_game(white, black, verbose=verbose)
                winner = result.winner()

                if winner == name_a:
                    a_wins += 1
                    wins[name_a] += 1
                    losses[name_b] += 1
                elif winner == name_b:
                    b_wins += 1
                    wins[name_b] += 1
                    losses[name_a] += 1
                else:
                    pair_draws += 1
                    draws[name_a] += 1
                    draws[name_b] += 1

            played = self.games_per_pair * 2
            print(
                f"    {name_a}: {a_wins}W / {pair_draws}D / {b_wins}L "
                f"({a_wins / played:.0%} WR)\n"
            )

        print("\n=== Krajnji rezultati ===")
        print(f"{'Rang':<5} {'Agent':<30} {'W':>5} {'D':>5} {'L':>5} {'WR':>7}")
        print("-" * 55)

        sorted_names = sorted(
            names,
            key=lambda n: wins[n] / max(1, wins[n] + losses[n] + draws[n]),
            reverse=True
        )
        for i, name in enumerate(sorted_names, 1):
            total_g = wins[name] + losses[name] + draws[name]
            wr = wins[name] / total_g if total_g > 0 else 0
            print(f"{i:<5} {name:<30} {wins[name]:>5} {draws[name]:>5} "
                  f"{losses[name]:>5} {wr:>7.1%}")

        return self._get_stats(wins, losses, draws)

    # Statistike i cuvanje

    def _get_stats(self, wins: dict, losses: dict, draws: dict) -> dict:
        names = list(agent_map.keys())
        stats = {"agents": {}, "games": []}

        for name in names:
            total_g = wins[name] + losses[name] + draws[name]
            stats["agents"][name] = {
                "wins": wins[name],
                "losses": losses[name],
                "draws": draws[name],
                "total": total_g,
                "win_rate": round(wins[name] / total_g, 3) if total_g > 0 else 0,
            }

        for r in self._results:
            stats["games"].append({
                "white": r.white_name,
                "black": r.black_name,
                "result": r.result,
                "moves": r.num_moves,
                "termination": r.termination,
                "white_avg_nodes": round(r.white_avg_nodes),
                "black_avg_nodes": round(r.black_avg_nodes),
                "white_avg_depth": round(r.white_avg_depth, 1),
                "black_avg_depth": round(r.black_avg_depth, 1),
                "white_avg_time": round(r.white_avg_time, 4),
                "black_avg_time": round(r.black_avg_time, 4),
            })

        return stats

    def save(self, path: str = "tournament_results.json"):
        names = list(agent_map.keys())
        wins = {n: 0 for n in names}
        losses = {n: 0 for n in names}
        draws = {n: 0 for n in names}

        for r in self._results:
            w = r.winner()
            if w == r.white_name:
                wins[r.white_name] += 1
                losses[r.black_name] += 1
            elif w == r.black_name:
                wins[r.black_name] += 1
                losses[r.white_name] += 1
            else:
                draws[r.white_name] += 1
                draws[r.black_name] += 1

        data = self._get_stats(wins, losses, draws)
        data["timestamp"] = datetime.now().isoformat()
        data["config"] = {
            "time_limit": self.time_limit,
            "games_per_pair": self.games_per_pair,
            "agents": names,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Rezultati sacuvani u: {path}")