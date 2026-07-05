import json
import os
from datetime import datetime
from typing import Callable, Optional

from search import PVSSearch, MCTSSearch, StockfishSearch
from state import HandCraftedState, StockfishState, NNUEState, make_stockfish_state, make_nnue_state
from game import Game
from game_result import GameResult

StockfishPretrained = make_stockfish_state(
    "./stockfish/stockfish-windows-x86-64-avx2.exe",
    "./models/nn-c288c895ea92.nnue"
)

StockfishTrained10E = make_stockfish_state(
    "./utils/easy_train_data/experiments/experiment_test/stockfish_base/src/stockfish.exe",
    "./models/model10.nnue",
)

StockfishTrained20E = make_stockfish_state(
    "./utils/easy_train_data/experiments/experiment_test/stockfish_base/src/stockfish.exe",
    "./models/model20.nnue",
)

# NNUE mreze -- stride MORA da odgovara onome sa kojim je mreza trenirana.
NNUE512 = make_nnue_state("./models/nnue_e4_b18000_mse0.016716.pt", stride=512)
NNUE641 = make_nnue_state("./models/nnue_e2_b32000_mse0.025056.pt", stride=641)


agent_map = {
    "Agent 1": (PVSSearch, HandCraftedState),
    "Agent 2(random)": (MCTSSearch, None),
    "Agent 2(eval)": (MCTSSearch, HandCraftedState),
    "Agent 2(nnue)": (MCTSSearch, NNUE512),
    "Agent 3(512S)": (PVSSearch, NNUE512),
    "Agent 3(641S)": (PVSSearch, NNUE641),
    "Agent 4(10E)": (PVSSearch, StockfishTrained10E),
    "Agent 4(20E)": (PVSSearch, StockfishTrained20E),
    "Agent 5": (PVSSearch, StockfishPretrained),
    "Stockfish": (StockfishSearch, None),
}

class Matchmaker:
    def __init__(self, time_limit: float = 1.0):
        """
        Args:
            time_limit: vremenski budzet po potezu u sekundama
        """
        self.time_limit = time_limit
        self._game = Game()
        self._results: list[GameResult] = []

    def play_game(
            self,
            white_name: str,
            black_name: str,
            move_callback: Optional[Callable] = None,
            verbose: bool = False,
    ) -> GameResult:
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

    @property
    def results(self) -> list[GameResult]:
        return self._results

    # Statistike i cuvanje

    def get_stats(self, wins: dict, losses: dict, draws: dict) -> dict:
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

    def save(self, path: str = "results/tournament_results.json", config: Optional[dict] = None):
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

        data = self.get_stats(wins, losses, draws)
        data["timestamp"] = datetime.now().isoformat()
        data["config"] = config or {"time_limit": self.time_limit, "agents": names}

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Rezultati sacuvani u: {path}")

