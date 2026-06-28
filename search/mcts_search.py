import time
import chess
import math
import random

from search import ChessSearch

C_SQRT_TWO = 1.414

class MCTSNode:
    """
    Cvor MCTS stabla
    w - ukupne pobede
    n - broj poseta cvoru
    """
    __slots__ = ["board", "parent", "move", "children", "w", "n", "_untried_moves"]
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []

        self.w: float = 0.0 # 1.0=beli 0.5=draw 0.0=crni
        self.n: int = 0

        self._untried_moves = list(self.board.legal_moves)

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

    def best_child(self, c: float = C_SQRT_TWO) -> "MCTSNode":
        return max(self.children, key=lambda node: node.ucb1(c))

class MCTSSearch(ChessSearch):
    def __init__(self, time_limit: float = 1.0, rollout_depth: int = 50, c: float = C_SQRT_TWO):
        super().__init__(time_limit)
        self.rollout_depth = rollout_depth
        self.c = c
        # Dodatna statisika za MCTS
        self.simulations_done = 0
        self.avg_rollout_depth = 0


    def best_move(self, board: chess.Board, state_class=None) -> chess.Move:
        self._reset_statistics()
        start = time.time()

        root = MCTSNode(board=board)

        while time.time() - start < self.time_limit:
            node: MCTSNode = self._select(root)
            node: MCTSNode = self._expand(node)
            result: float = self._simulate(node)
            self._backpropagate(node, result)
            self.simulations_done += 1

        self.nodes_visited = root.n
        self.time_elapsed = time.time() - start

        # Biranje poteza sa najvise poseta (UBC1 ga video kao najobecavajuceg)
        if not root.children:
            return list(board.legal_moves)[0]

        best: MCTSNode = max(root.children, key=lambda n: n.n)

        return best.move


    def _select(self, node : MCTSNode) -> MCTSNode:
        """
        Faza 1 - Selekcija
        Silazimo niz stablo birajuci cvorove po UCB1 heuristici
        dok ne dodjemo do neistrazenog cvora ili kraja igre
        """
        max_selection_depth = 50
        current_selection_depth = 0
        while not node.board.is_game_over() and current_selection_depth < max_selection_depth:
            if not node.is_fully_expanded():
                return node
            if not node.children:
                break
            node = node.best_child(self.c)
            current_selection_depth += 1

        if current_selection_depth > self.depth_reached:
            self.depth_reached = current_selection_depth

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Faza 2 - Ekspanzija
        Dodajemo jedan novi cvor za neistrazeni potez
        """
        if node.board.is_game_over():
            return node

        untried: list = node.untried_moves
        for move in untried:
            if node.board.gives_check(move):
                test_board = node.board
                test_board.push(move)
                if test_board.is_checkmate():
                    untried.remove(move)
                    child = MCTSNode(board=test_board, parent=node, move=move)
                    node.children.append(child)
                    test_board.pop()
                    return child
                test_board.pop()

        move = random.choice(untried)
        node.untried_moves.remove(move)

        new_board = node.board.copy()
        new_board.push(move)

        child = MCTSNode(board=new_board, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        Faza 3 - Simulacija (Rollout)
        Igramo nasumicno do kraja pratije ili max dubine
        Umesto Centipawn evalucije, postoji samo ishod
        Returns:
            1.0 = pobeda belog
            0.0 = pobeda crnog
            0.5 = remi
        """
        board = node.board
        moves_made = []
        current_depth = 0

        while not board.is_game_over() and current_depth < self.rollout_depth:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            captures = [m for m in legal_moves if board.is_capture(m)]

            if captures:
                if random.random() < 0.7:
                    move = random.choice(captures)
                else:
                    move = random.choice(legal_moves)
            else:
                move = random.choice(legal_moves)

            board.push(move)
            moves_made.append(move)
            current_depth += 1

        self.avg_rollout_depth = (
                (self.avg_rollout_depth * self.simulations_done + current_depth)
                / (self.simulations_done + 1)
        )

        for _ in moves_made:
            board.pop()

        return self._outcome(board)

    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Faza 4 - Propagacija
        Rezultat se salje unazad kroz sve roditelje
        """
        while node is not None:
            node.n += 1
            if node.board.turn == chess.BLACK:
                node.w += result
            else:
                node.w += 1.0 - result
            node = node.parent

    def _outcome(self, board: chess.Board) -> float:
        if not board.is_game_over():
            return 0.5

        result = board.result()
        if result == "1-0": return 1.0
        elif result == "0-1": return 0.0

        return 0.5

    # STATISTICS
    def _reset_statistics(self):
        super()._reset_statistics()
        self.simulations_done = 0
        self.avg_rollout_depth = 0

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats.update({
            "simulations": self.simulations_done,
            "avg_rollout_depth": self.avg_rollout_depth,
        })
        return stats