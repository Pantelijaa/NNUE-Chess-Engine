import chess
import math

C_SQRT_TWO = 1.414

class MCTSNode:
    """
    Cvor MCTS stabla
    w - ukupne pobede
    n - broj poseta cvoru
    """
    __slots__ = ["board", "parent", "move", "children", "w", "n", "_untried_moves"]
    def __init__(self, board: chess.Board, parent=None, move: chess.Move | None = None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []

        self.w: float = 0.0 # 1.0=beli 0.5=draw 0.0=crni
        self.n: int = 0

        self._untried_moves = None

        @property
        def untried_moves(self):
            if self._untried_moves is None:
                tried = {child.move for child in self.children}
                self._untried_moves = [m for m in self.board.legal_moves if m not in tried]

            return self._untried_moves

        def is_fully_expanded(self) -> bool:
            return len(self.untried_moves) == 0

        def win_rate(self) -> float:
            return self.w / self.n if self.n > 0 else 0.0

        def ucb1(self, c: float = C_SQRT_TWO) -> float:
            if self.n == 0:
                return float("inf")

            exploitation = self.w / self.n
            exploration = c * math.sqrt(math.log(self.parent.n) / self.n)
            return exploitation + exploration

        def best_child(self, c: float = C_SQRT_TWO) -> chess.Move:
            return max(self.children, key=lambda node: node.ucb1(c))