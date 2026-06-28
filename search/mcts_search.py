import time
import chess
import math
import random

from search import ChessSearch

C_SQRT_TWO = 1.414
EVAL_SCALE = 600.0


class MCTSNode:
    __slots__ = ["move", "turn", "parent", "children", "w", "n", "untried_moves"]

    def __init__(self, move=None, parent=None, turn=chess.WHITE):
        self.move = move
        self.parent = parent
        self.children = []
        self.w = 0.0
        self.n = 0
        self.turn = turn
        self.untried_moves = None

    def win_rate(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0

    def ucb1(self, parent_n: int, c: float = C_SQRT_TWO) -> float:
        if self.n == 0:
            return float("inf")
        return self.w / self.n + c * math.sqrt(math.log(parent_n) / self.n)

    def best_child(self, c: float = C_SQRT_TWO) -> "MCTSNode":
        pn = self.n
        return max(self.children, key=lambda node: node.ucb1(pn, c))


class MCTSSearch(ChessSearch):
    def __init__(self, time_limit: float = 1.0, c: float = C_SQRT_TWO):
        super().__init__(time_limit)
        self.c = c
        self.simulations_done = 0

    def best_move(self, board: chess.Board, state_class=None) -> chess.Move:
        self._reset_statistics()
        if state_class is not None:
            state_class.clear_eval_cache()
        start = time.time()

        root = MCTSNode(turn=board.turn)
        root.untried_moves = list(board.legal_moves)

        while time.time() - start < self.time_limit:
            node, depth_pushed = self._select_expand(root, board)
            if state_class is not None:
                result = self._evaluate(board, state_class)
            else:
                result = self._simulate(board)
            self._backpropagate(node, result)

            for _ in range(depth_pushed):
                board.pop()

            self.simulations_done += 1

        self.nodes_visited = root.n
        self.time_elapsed = time.time() - start

        if not root.children:
            moves = list(board.legal_moves)
            return moves[0] if moves else chess.Move.null()

        best = max(root.children, key=lambda n: n.n)
        return best.move

    def _select_expand(self, root: MCTSNode, board: chess.Board):
        node = root
        depth_pushed = 0

        while True:
            if board.is_game_over():
                return node, depth_pushed

            if node.untried_moves is None:
                node.untried_moves = list(board.legal_moves)

            if node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                move = node.untried_moves[idx]
                node.untried_moves[idx] = node.untried_moves[-1]
                node.untried_moves.pop()

                board.push(move)
                depth_pushed += 1

                child = MCTSNode(move=move, parent=node, turn=board.turn)
                node.children.append(child)

                if depth_pushed > self.depth_reached:
                    self.depth_reached = depth_pushed

                return child, depth_pushed

            if not node.children:
                return node, depth_pushed

            node = node.best_child(self.c)
            board.push(node.move)
            depth_pushed += 1

            if depth_pushed > self.depth_reached:
                self.depth_reached = depth_pushed

    def _evaluate(self, board: chess.Board, state_class) -> float:
        if board.is_game_over():
            return self._outcome(board)

        state = state_class()
        score_stm = state.get_eval_score(board)
        white_score = score_stm if board.turn == chess.WHITE else -score_stm
        return 1.0 / (1.0 + math.exp(-white_score / EVAL_SCALE))

    def _simulate(self, board: chess.Board) -> float:
        moves_made = 0
        rollout_depth = 20

        while not board.is_game_over() and moves_made < rollout_depth:
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
            moves_made += 1

        result = self._outcome(board)

        for _ in range(moves_made):
            board.pop()

        return result

    def _backpropagate(self, node: MCTSNode, result: float):
        while node is not None:
            node.n += 1
            if node.turn == chess.BLACK:
                node.w += result
            else:
                node.w += 1.0 - result
            node = node.parent

    def _outcome(self, board: chess.Board) -> float:
        if not board.is_game_over():
            return 0.5
        result = board.result()
        if result == "1-0": return 1.0
        if result == "0-1": return 0.0
        return 0.5

    def _reset_statistics(self):
        super()._reset_statistics()
        self.simulations_done = 0

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats.update({"simulations": self.simulations_done})
        return stats
