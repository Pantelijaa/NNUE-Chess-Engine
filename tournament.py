"""
Turnir bez GUI-ja.
"""

import win_startup  # noqa: F401  # mora pre torch importa (i u paralelnim radnicima)

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import numpy as np
import matplotlib.pyplot as plt

from matchmaker import Matchmaker, agent_map
from game_result import GameResult

tournament_map = [
    ("Stockfish", "PVSPretrainedStockfish"),
    ("PVSPretrainedStockfish", "PVSStockfish"),
    ("NNUE", "PVSPretrainedStockfish"),
    ("NNUE", "PVSStockfish"),
    ("NNUE", "Handcrafted"),
    ("Handcrafted", "MCTS_eval"),
    ("Handcrafted", "MCTS_random"),
]


def _validated_pairs():
    pairs = [tuple(p) for p in tournament_map]
    if not pairs:
        raise ValueError("tournament_map je prazan.")
    for p in pairs:
        if len(p) != 2:
            raise ValueError(f"tournament_map: par mora imati tacno 2 agenta: {p}")
        for n in p:
            if n not in agent_map:
                raise ValueError(
                    f"tournament_map: agent '{n}' ne postoji u agent_map "
                    f"(dostupni: {list(agent_map)})."
                )
    return pairs


def _participants(pairs):
    seen = []
    for a, b in pairs:
        for n in (a, b):
            if n not in seen:
                seen.append(n)
    return seen


def _worker_init(sf_threads: int, sf_hash: int):
    os.environ["STOCKFISH_THREADS"] = str(sf_threads)
    os.environ["STOCKFISH_HASH"] = str(sf_hash)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _close_engines():
    """Gasi Stockfish motore (agent i eval-state) i nulira singletone.

    Kljucno za paralelni rezim: chess.engine pokrece NE-daemon nit po motoru,
    pa otvoreni motor sprecava izlazak radnog procesa (program se ne zavrsava).
    Gasenjem na kraju svake partije nit se zatvara, a proces moze cisto da izadje.
    """
    try:
        from search.stockfish_search import StockfishSearch
        if StockfishSearch._engine is not None:
            try:
                StockfishSearch._engine.quit()
            except Exception:
                pass
            StockfishSearch._engine = None
    except Exception:
        pass
    try:
        from state.stockfish_state import StockfishState
        StockfishState.close()  # zatvara motore svih (binary, nnue) konfiguracija
    except Exception:
        pass


def _play_job(job: tuple) -> dict:
    """Odigrava jednu partiju u zasebnom procesu i vraca lagani (picklable) rezultat.
    Greske (npr. pad Stockfish procesa) se hvataju da ne obore ceo turnir."""
    white_name, black_name, pair_a, pair_b, time_limit = job
    try:
        mm = Matchmaker(time_limit=time_limit)
        r = mm.play_game(white_name, black_name)  # bez callbacka u paralelnom rezimu
        return {
            "ok": True,
            "pair_a": pair_a, "pair_b": pair_b,
            "white_name": r.white_name, "black_name": r.black_name,
            "result": r.result, "num_moves": r.num_moves, "termination": r.termination,
            "white_avg_nodes": r.white_avg_nodes, "black_avg_nodes": r.black_avg_nodes,
            "white_avg_depth": r.white_avg_depth, "black_avg_depth": r.black_avg_depth,
            "white_avg_time": r.white_avg_time, "black_avg_time": r.black_avg_time,
        }
    except Exception as e:
        return {
            "ok": False, "pair_a": pair_a, "pair_b": pair_b,
            "white_name": white_name, "black_name": black_name,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        # Zatvori motore izmedju partija: cisti izlazak procesa + oslobadja memoriju.
        _close_engines()


class Tournament:
    def __init__(self, time_limit: float = 1.0, games_per_side: int = 20):
        """
        Args:
            time_limit:     vremenski budzet po potezu u sekundama
            games_per_side: broj partija po strani za svaki par
                            (svaki agent odigra toliko kao beli i toliko kao crni)
        """
        self.time_limit = time_limit
        self.games_per_side = games_per_side
        self.matchmaker = Matchmaker(time_limit=time_limit)

    @staticmethod
    def _print_state(board: chess.Board, move: chess.Move):
        """Callback koji ispisuje tablu posle svakog poteza (zamena za GUI)."""
        mover = "Crni" if board.turn == chess.WHITE else "Beli"
        print(f"\n{mover} odigrao {move.uci()} (potez {board.fullmove_number}):")

    def run(self, verbose: bool = True) -> dict:
        pairs = _validated_pairs()
        names = _participants(pairs)
        games_per_pair = self.games_per_side * 2
        total = len(pairs) * games_per_pair

        print(
            f"Turnir pocinje: {len(names)} agenata, {len(pairs)} parova, "
            f"{games_per_pair} partija po paru, {total} partija ukupno\n"
        )

        wins = {name: 0 for name in names}
        losses = {name: 0 for name in names}
        draws = {name: 0 for name in names}

        callback = self._print_state if verbose else None

        for name_a, name_b in pairs:
            print(f"\n{'=' * 60}\n  {name_a}  vs  {name_b}\n{'=' * 60}")
            a_wins = b_wins = pair_draws = 0

            for game_num in range(games_per_pair):
                if game_num < self.games_per_side:
                    white, black = name_a, name_b
                else:
                    white, black = name_b, name_a

                print(f"\n  Partija {game_num + 1}/{games_per_pair}: "
                      f"{white} (beli) vs {black} (crni)")

                result = self.matchmaker.play_game(
                    white, black, move_callback=callback
                )
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

                print(f"  -> {result.result} ({result.termination}, "
                      f"{result.num_moves} poteza) | "
                      f"stanje para: {name_a} {a_wins}-{pair_draws}-{b_wins} {name_b}")

            print(
                f"\n  Rezultat para — {name_a}: {a_wins}W / {pair_draws}D / {b_wins}L "
                f"({a_wins / games_per_pair:.0%} WR)"
            )

        return self._finalize(names, wins, losses, draws)

    def run_parallel(self, workers: int, sf_threads: int = 1, sf_hash: int = 16,
                     verbose: bool = True) -> dict:
        pairs = _validated_pairs()
        names = _participants(pairs)

        jobs = []
        for name_a, name_b in pairs:
            for _ in range(self.games_per_side):
                jobs.append((name_a, name_b, name_a, name_b, self.time_limit))  # a beli
            for _ in range(self.games_per_side):
                jobs.append((name_b, name_a, name_a, name_b, self.time_limit))  # a crni

        total = len(jobs)
        print(
            f"Turnir (paralelno, {workers} procesa, Stockfish {sf_threads} niti / "
            f"{sf_hash} MB) pocinje: {len(names)} agenata, {len(pairs)} parova, "
            f"{total} partija ukupno\n"
        )

        wins = {name: 0 for name in names}
        losses = {name: 0 for name in names}
        draws = {name: 0 for name in names}

        completed = 0
        failed = 0
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init, initargs=(sf_threads, sf_hash)
        ) as ex:
            futures = [ex.submit(_play_job, job) for job in jobs]
            for fut in as_completed(futures):
                completed += 1
                try:
                    res = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"  [{completed:>4}/{total}] GRESKA (proces): {e}")
                    continue

                if not res.get("ok", True):
                    failed += 1
                    print(f"  [{completed:>4}/{total}] PARTIJA PALA: "
                          f"{res['white_name']} vs {res['black_name']} — {res['error']}")
                    continue

                gr = self._result_from_dict(res)
                self.matchmaker._results.append(gr)

                name_a, name_b = res["pair_a"], res["pair_b"]
                winner = gr.winner()
                if winner == name_a:
                    wins[name_a] += 1
                    losses[name_b] += 1
                elif winner == name_b:
                    wins[name_b] += 1
                    losses[name_a] += 1
                else:
                    draws[name_a] += 1
                    draws[name_b] += 1

                if verbose:
                    print(
                        f"  [{completed:>3}/{total}] {res['white_name']} (beli) vs "
                        f"{res['black_name']} (crni) -> {res['result']} "
                        f"({res['termination']}, {res['num_moves']} poteza)"
                    )

        if failed:
            print(f"\nUPOZORENJE: {failed}/{total} partija nije zavrseno "
                  f"(verovatno nedostatak memorije — smanji --workers ili --sf-hash).")

        return self._finalize(names, wins, losses, draws)

    @staticmethod
    def _result_from_dict(res: dict) -> GameResult:
        gr = GameResult(
            white_name=res["white_name"], black_name=res["black_name"],
            result=res["result"], num_moves=res["num_moves"],
            termination=res["termination"],
        )
        gr.white_avg_nodes = res["white_avg_nodes"]
        gr.black_avg_nodes = res["black_avg_nodes"]
        gr.white_avg_depth = res["white_avg_depth"]
        gr.black_avg_depth = res["black_avg_depth"]
        gr.white_avg_time = res["white_avg_time"]
        gr.black_avg_time = res["black_avg_time"]
        return gr

    def _finalize(self, names, wins, losses, draws) -> dict:
        self._print_standings(names, wins, losses, draws)

        stats = self.matchmaker.get_stats(wins, losses, draws)
        self.matchmaker.save(
            path="results/tournament_results.json",
            config={
                "format": "consecutive_pairs",
                "time_limit": self.time_limit,
                "games_per_side": self.games_per_side,
                "agents": names,
            },
        )
        self.plot(names, wins, losses, draws)
        return stats

    def _print_standings(self, names, wins, losses, draws):
        print("\n\n=== Krajnji rezultati ===")
        print(f"{'Rang':<5} {'Agent':<15} {'W':>5} {'D':>5} {'L':>5} {'WR':>8}")
        print("-" * 46)

        sorted_names = sorted(
            names,
            key=lambda n: wins[n] / max(1, wins[n] + losses[n] + draws[n]),
            reverse=True,
        )
        for i, name in enumerate(sorted_names, 1):
            total_g = wins[name] + losses[name] + draws[name]
            wr = wins[name] / total_g if total_g > 0 else 0
            print(f"{i:<5} {name:<15} {wins[name]:>5} {draws[name]:>5} "
                  f"{losses[name]:>5} {wr:>8.1%}")

    def _agent_search_stats(self, names):
        """Agregira info o agentu iz get_statistics (preko GameResult proseka)."""
        nodes = {n: [] for n in names}
        depth = {n: [] for n in names}
        time_ = {n: [] for n in names}

        for r in self.matchmaker.results:
            nodes[r.white_name].append(r.white_avg_nodes)
            depth[r.white_name].append(r.white_avg_depth)
            time_[r.white_name].append(r.white_avg_time)
            nodes[r.black_name].append(r.black_avg_nodes)
            depth[r.black_name].append(r.black_avg_depth)
            time_[r.black_name].append(r.black_avg_time)

        return (
            {n: self.mean(nodes[n]) for n in names},
            {n: self.mean(depth[n]) for n in names},
            {n: self.mean(time_[n]) for n in names},
        )

    @staticmethod
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def plot(self, names, wins, losses, draws, out_dir: str = "results"):
        """Svaki grafikon se cuva kao zaseban fajl u out_dir."""
        os.makedirs(out_dir, exist_ok=True)

        self._plot_wdl(names, wins, losses, draws,
                       os.path.join(out_dir, "plot_wdl.png"))
        self._plot_winrate_by_opponent(names,
                                       os.path.join(out_dir, "plot_winrate.png"))

        avg_nodes, avg_depth, avg_time = self._agent_search_stats(names)
        self._plot_metric(names, avg_nodes, "Prosecan broj cvorova po potezu",
                          "nodes_visited", os.path.join(out_dir, "plot_nodes.png"),
                          color="#8e44ad", log=True, fmt="{:,.0f}")
        self._plot_metric(names, avg_depth, "Prosecna dubina po potezu",
                          "depth_reached", os.path.join(out_dir, "plot_depth.png"),
                          color="#e67e22", fmt="{:.1f}")
        self._plot_metric(names, avg_time, "Prosecno vreme po potezu",
                          "time_elapsed (s)", os.path.join(out_dir, "plot_time.png"),
                          color="#16a085", fmt="{:.3f}")
        plt.show()

    def _plot_wdl(self, names, wins, losses, draws, path):
        w = [wins[n] for n in names]
        d = [draws[n] for n in names]
        l = [losses[n] for n in names]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, w, label="Pobede", color="#45a049")
        ax.bar(names, d, bottom=w, label="Remiji", color="#b5b3b1")
        ax.bar(names, l, bottom=[wi + di for wi, di in zip(w, d)],
               label="Porazi", color="#c0392b")
        ax.set_title("Pobede / Remiji / Porazi (ukupno)")
        ax.set_ylabel("Broj partija")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        print(f"Grafikon sacuvan: {path}")

    def _head_to_head(self, names):
        """Rezultati po paru (head-to-head), levi = raniji u agent_map."""
        order = {n: i for i, n in enumerate(names)}
        agg = {}
        for r in self.matchmaker.results:
            key = tuple(sorted((r.white_name, r.black_name),
                               key=lambda n: order.get(n, 0)))
            e = agg.setdefault(key, {"a": key[0], "b": key[1], "a_wins": 0,
                                     "draws": 0, "b_wins": 0, "games": 0, "moves": 0})
            e["games"] += 1
            e["moves"] += r.num_moves
            w = r.winner()
            if w == "draw":
                e["draws"] += 1
            elif w == key[0]:
                e["a_wins"] += 1
            else:
                e["b_wins"] += 1
        # sortiraj po redosledu iz agent_map radi stabilnog prikaza
        return sorted((e for e in agg.values() if e["games"] > 0),
                      key=lambda e: (order.get(e["a"], 0), order.get(e["b"], 0)))

    def _plot_winrate_by_opponent(self, names, path):
        matchups = self._head_to_head(names)
        if not matchups:
            return
        labels = [f"{m['a']}\nvs\n{m['b']}\n(~{m['moves'] / m['games']:.0f} poteza)"
                  for m in matchups]
        a_wr = [m["a_wins"] / m["games"] * 100 for m in matchups]
        dr = [m["draws"] / m["games"] * 100 for m in matchups]
        b_wr = [m["b_wins"] / m["games"] * 100 for m in matchups]

        x = np.arange(len(matchups))
        width = 0.27
        fig, ax = plt.subplots(figsize=(max(8, 2.4 * len(matchups)), 6))
        bars = [
            ax.bar(x - width, a_wr, width, label="Pobeda prvog", color="#45a049"),
            ax.bar(x, dr, width, label="Remi", color="#b5b3b1"),
            ax.bar(x + width, b_wr, width, label="Pobeda drugog", color="#c0392b"),
        ]
        for b in bars:
            ax.bar_label(b, fmt="%.0f%%", padding=2, fontsize=8)
        ax.set_title("Procenat pobeda po protivniku (head-to-head)")
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        print(f"Grafikon sacuvan: {path}")

    def _plot_metric(self, names, values, title, ylabel, path,
                     color="#2980b9", log=False, fmt="{:.1f}"):
        vals = [values[n] for n in names]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, vals, color=color)
        use_log = log and min(vals) > 0
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], padding=3, fontsize=9)
        top = ax.get_ylim()[1]
        ax.set_ylim(top=top * (2.5 if use_log else 1.15))  # prostor za labele
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        print(f"Grafikon sacuvan: {path}")
