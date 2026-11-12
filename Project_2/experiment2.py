"""当代人工智能实验二：A*、Minimax 与 Alpha-Beta 对照实验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import redirect_stdout
from heapq import heappop, heappush
from io import StringIO
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

Coord = tuple[int, int]


@dataclass
class SearchStats:
    expanded: int = 0
    leaves: int = 0
    pruned: int = 0
    elapsed_ms: float = 0.0


def _neighbors(pos: Coord, rows: int, cols: int) -> Iterable[Coord]:
    r, c = pos
    for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols:
            yield nxt


def astar_path(
    grid: list[list[Any]], start: Coord, goal: Coord,
    passable: Callable[[Any], bool] | None = None,
    enter_cost: Callable[[Any], float] | None = None,
    heuristic: Callable[[Coord, Coord], float] | None = None,
) -> tuple[list[Coord], float, SearchStats]:
    """A* 搜索；返回路径、总代价和搜索统计。"""
    rows, cols = len(grid), len(grid[0])
    passable = passable or (lambda cell: cell != "X")
    enter_cost = enter_cost or (lambda _cell: 1)
    heuristic = heuristic or (lambda _a, _b: 0)
    stats = SearchStats()
    if not (0 <= start[0] < rows and 0 <= start[1] < cols
            and 0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return [], float("inf"), stats
    if not passable(grid[start[0]][start[1]]) or not passable(grid[goal[0]][goal[1]]):
        return [], float("inf"), stats

    serial = count()
    frontier = [(heuristic(start, goal), next(serial), 0.0, start)]
    best_cost = {start: 0.0}
    parent: dict[Coord, Coord] = {}
    began = perf_counter()
    while frontier:
        _, _, cost, here = heappop(frontier)
        if cost != best_cost.get(here):
            continue
        stats.expanded += 1
        if here == goal:
            path = [here]
            while here in parent:
                here = parent[here]
                path.append(here)
            stats.elapsed_ms = (perf_counter() - began) * 1000
            return path[::-1], cost, stats
        for nxt in _neighbors(here, rows, cols):
            cell = grid[nxt[0]][nxt[1]]
            if not passable(cell):
                continue
            new_cost = cost + enter_cost(cell)
            if new_cost < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = new_cost
                parent[nxt] = here
                priority = new_cost + heuristic(nxt, goal)
                heappush(frontier, (priority, next(serial), new_cost, nxt))
    stats.elapsed_ms = (perf_counter() - began) * 1000
    return [], float("inf"), stats


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def maze_experiment() -> None:
    grid = [list(row) for row in ("S..X.", ".X...", ".X.X.", "...XE", "XX...")]
    start, goal = (0, 0), (3, 4)
    for label, h in (("h=0", lambda _a, _b: 0), ("曼哈顿", manhattan)):
        path, cost, stats = astar_path(grid, start, goal, heuristic=h)
        print(f"迷宫 A* [{label}]: path={path}, length={int(cost) if path else -1}, "
              f"expanded={stats.expanded}, time={stats.elapsed_ms:.3f}ms")


def weighted_experiment() -> None:
    grid: list[list[Any]] = [
        ["S", 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1], [8, 8, 1, 1, "E"],
    ]
    start, goal = (0, 0), (4, 4)
    # E 的进入代价按题目样例取 0，故启发式只估算抵达 E 前的步数。
    h = lambda a, b: max(0, manhattan(a, b) - 1)
    for label, heuristic in (("h=0", lambda _a, _b: 0), ("加权曼哈顿", h)):
        path, cost, stats = astar_path(
            grid, start, goal,
            enter_cost=lambda cell: 0 if cell == "E" else (1 if cell == "S" else cell),
            heuristic=heuristic,
        )
        print(f"加权地形 A* [{label}]: path={path}, cost={int(cost) if path else -1}, "
              f"expanded={stats.expanded}, time={stats.elapsed_ms:.3f}ms")


@dataclass
class TreeNode:
    value: float | None = None
    children: list["TreeNode"] = field(default_factory=list)


def minimax(node: TreeNode, maximizing: bool = True,
            stats: SearchStats | None = None) -> float:
    stats = stats if stats is not None else SearchStats()
    if not node.children:
        if node.value is None:
            raise ValueError("叶节点必须提供 value")
        stats.leaves += 1
        return node.value
    stats.expanded += 1
    values = [minimax(child, not maximizing, stats) for child in node.children]
    return max(values) if maximizing else min(values)


def alpha_beta(node: TreeNode, maximizing: bool = True,
               alpha: float = float("-inf"), beta: float = float("inf"),
               stats: SearchStats | None = None) -> float:
    stats = stats if stats is not None else SearchStats()
    if not node.children:
        if node.value is None:
            raise ValueError("叶节点必须提供 value")
        stats.leaves += 1
        return node.value
    stats.expanded += 1
    if maximizing:
        value = float("-inf")
        for i, child in enumerate(node.children):
            value = max(value, alpha_beta(child, False, alpha, beta, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.pruned += len(node.children) - i - 1
                break
        return value
    value = float("inf")
    for i, child in enumerate(node.children):
        value = min(value, alpha_beta(child, True, alpha, beta, stats))
        beta = min(beta, value)
        if alpha >= beta:
            stats.pruned += len(node.children) - i - 1
            break
    return value


def _sample_tree() -> TreeNode:
    leaf = lambda n: TreeNode(value=n)
    max1 = TreeNode(children=[leaf(3), leaf(5)])
    max2 = TreeNode(children=[leaf(6), leaf(9)])
    max3 = TreeNode(children=[leaf(1), leaf(2)])
    return TreeNode(children=[
        TreeNode(children=[max1, max2]),
        TreeNode(children=[max2, max3]),
    ])


def game_tree_experiment() -> None:
    for order, reverse in (("从左到右", False), ("反向访问", True)):
        for name, solver in (("Minimax", minimax), ("Alpha-Beta", alpha_beta)):
            tree = _sample_tree()
            if reverse:
                tree.children.reverse()
                for child in tree.children:
                    child.children.reverse()
                    for grandchild in child.children:
                        grandchild.children.reverse()
            stats = SearchStats()
            began = perf_counter()
            value = solver(tree, stats=stats)
            stats.elapsed_ms = (perf_counter() - began) * 1000
            print(f"博弈树 [{order}] {name}: value={value:g}, leaves={stats.leaves}, "
                  f"pruned={stats.pruned}, time={stats.elapsed_ms:.3f}ms")


TREASURE_GRID: list[list[Any]] = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
TREASURE_POS: dict[str, Coord] = {
    "A": (0, 0), "B": (6, 6), "T1": (0, 6),
    "T2": (2, 2), "T3": (4, 4), "T4": (6, 0),
}
# 按题目预期结果校准并与价值表核对：T1=9、T2=6、T3=7、T4=7。
TREASURE_VALUE = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


def treasure_distances() -> dict[tuple[str, str], int]:
    names = list(TREASURE_POS)
    distances: dict[tuple[str, str], int] = {}
    for source in names:
        for target in names:
            _, cost, _ = astar_path(
                TREASURE_GRID, TREASURE_POS[source], TREASURE_POS[target],
                enter_cost=lambda cell: 1 if isinstance(cell, str) else cell,
                heuristic=lambda _a, _b: 0,
            )
            distances[source, target] = int(cost) if cost != float("inf") else -1
    return distances


@dataclass(frozen=True)
class GameState:
    a_pos: str
    b_pos: str
    remaining: tuple[str, ...]
    score_a: int = 0
    score_b: int = 0
    maximizing: bool = True


def _ordered_targets(state: GameState, distances: dict[tuple[str, str], int],
                     order: bool) -> list[str]:
    targets = list(state.remaining)
    if order:
        pos = state.a_pos if state.maximizing else state.b_pos
        targets.sort(key=lambda t: TREASURE_VALUE[t] - distances[pos, t], reverse=state.maximizing)
    return targets


def treasure_search(distances: dict[tuple[str, str], int], depth: int,
                    use_alpha_beta: bool, ordered: bool = False) -> tuple[str | None, int, SearchStats]:
    stats = SearchStats()
    began = perf_counter()

    def visit(state: GameState, left: int, alpha: float, beta: float) -> tuple[int, str | None]:
        stats.expanded += 1
        if left == 0 or not state.remaining:
            stats.leaves += 1
            return state.score_a - state.score_b, None
        best = float("-inf") if state.maximizing else float("inf")
        best_target = None
        choices = _ordered_targets(state, distances, ordered)
        for i, target in enumerate(choices):
            current = state.a_pos if state.maximizing else state.b_pos
            distance = distances[current, target]
            gain = TREASURE_VALUE[target] - distance
            next_state = GameState(
                a_pos=target if state.maximizing else state.a_pos,
                b_pos=target if not state.maximizing else state.b_pos,
                remaining=tuple(t for t in state.remaining if t != target),
                score_a=state.score_a + (gain if state.maximizing else 0),
                score_b=state.score_b + (gain if not state.maximizing else 0),
                maximizing=not state.maximizing,
            )
            value, _ = visit(next_state, left - 1, alpha, beta)
            if (state.maximizing and value > best) or (not state.maximizing and value < best):
                best, best_target = value, target
            if use_alpha_beta:
                if state.maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    stats.pruned += len(choices) - i - 1
                    break
        return int(best), best_target

    initial = GameState("A", "B", tuple(TREASURE_VALUE))
    value, target = visit(initial, depth, float("-inf"), float("inf"))
    stats.elapsed_ms = (perf_counter() - began) * 1000
    return target, value, stats


def treasure_experiment() -> None:
    distances = treasure_distances()
    names = list(TREASURE_POS)
    print("\n宝藏位置最小代价矩阵（行/列顺序：" + ", ".join(names) + "):")
    print("     " + " ".join(f"{n:>4}" for n in names))
    for source in names:
        print(f"{source:>4} " + " ".join(f"{distances[source, t]:>4}" for t in names))
    for depth in (2, 4):
        print(f"H={depth}")
        for name, use_ab, ordered in (
            ("Minimax", False, False), ("Alpha-Beta", True, False),
            ("Alpha-Beta（估值排序）", True, True),
        ):
            target, value, stats = treasure_search(distances, depth, use_ab, ordered)
            print(f"  {name}: best_target={target}, best_value={value}, "
                  f"nodes={stats.expanded}, leaves={stats.leaves}, pruned={stats.pruned}, "
                  f"time={stats.elapsed_ms:.3f}ms")


def _run_experiments() -> None:
    maze_experiment()
    weighted_experiment()
    game_tree_experiment()
    treasure_experiment()


def main() -> None:
    output = StringIO()
    with redirect_stdout(output):
        _run_experiments()
    result_text = output.getvalue()
    print(result_text, end="")
    result_path = Path(__file__).with_name("experiment2_results.txt")
    result_path.write_text(result_text, encoding="utf-8")
    print(f"\n结果已保存到：{result_path}")


if __name__ == "__main__":
    main()
