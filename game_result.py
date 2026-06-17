from dataclasses import dataclass, field


@dataclass
class GameResult:
    """
    Cuva sve relevantne informacije o jednoj odigranoj partiji.
    Koristi se u Tournament klasi za agregaciju statistika.
    """
    white_name:  str
    black_name:  str
    result:      str   # "1-0", "0-1", "1/2-1/2"
    num_moves:   int   # broj polupoteza
    termination: str   # "checkmate", "stalemate", "move_limit"...

    # Statistike pretrage po potezu za svaku stranu
    white_stats: list = field(default_factory=list)
    black_stats: list = field(default_factory=list)

    # Proseci - popunjavaju se pozivom compute_averages()
    white_avg_nodes: float = 0.0
    black_avg_nodes: float = 0.0
    white_avg_time:  float = 0.0
    black_avg_time:  float = 0.0
    white_avg_depth: float = 0.0
    black_avg_depth: float = 0.0

    def compute_averages(self):
        """Izracunava prosecne statistike na kraju partije."""
        if self.white_stats:
            self.white_avg_nodes = sum(s["nodes_visited"] for s in self.white_stats) / len(self.white_stats)
            self.white_avg_time  = sum(s["time_elapsed"]  for s in self.white_stats) / len(self.white_stats)
            self.white_avg_depth = sum(s["depth_reached"] for s in self.white_stats) / len(self.white_stats)
        if self.black_stats:
            self.black_avg_nodes = sum(s["nodes_visited"] for s in self.black_stats) / len(self.black_stats)
            self.black_avg_time  = sum(s["time_elapsed"]  for s in self.black_stats) / len(self.black_stats)
            self.black_avg_depth = sum(s["depth_reached"] for s in self.black_stats) / len(self.black_stats)

    def winner(self) -> str:
        """Vraca ime pobednika ili 'draw'."""
        if self.result == "1-0":   return self.white_name
        if self.result == "0-1":   return self.black_name
        return "draw"

    def __repr__(self):
        return (
            f"GameResult({self.white_name} vs {self.black_name}: "
            f"{self.result}, {self.num_moves} poteza, {self.termination})"
        )
