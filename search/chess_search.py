from abc import ABC, abstractmethod
import chess
import time

class ChessSearch(ABC):
    """
    Apstraktna klasa za algoritme pretrage
    """

    def __init__(self, time_limit: float = 2.0):
        self.time_limit = time_limit
        # Pracenje statistike pretrage
        self.nodes_visited: int = 0
        self.time_elapsed: float = 0.0
        self.depth_reached: int = 0

    @abstractmethod
    def best_move(self, board: chess.Board, state_class) -> chess.Move:
        """
        Pronalazi najbolji moguci potez u trenutnoj poziciji
        :param board1: trenutno stanje table
        :param state_class: klasa stanja koje se evaluira
        :return: Najbolja pronadjena pozicija
        """
        pass

    def _reset_statistics(self):
        self.nodes_visited: int = 0
        self.time_elapsed: float = 0.0
        self.depth_reached: int = 0

    def _get_statistics(self) -> dict:
        return {
            "time_elapsed": self.time_elapsed,
            "time_limit": self.time_limit,
            "nodes_visited": self.nodes_visited,
            "depth_reached": self.depth_reached,
        }
