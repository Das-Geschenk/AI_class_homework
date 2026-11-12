"""实验二选做拓展：Alpha-Beta 节点排序与自设迷宫 A* 对照。"""

from __future__ import annotations

import csv
import random
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from statistics import mean
from time import perf_counter

from experiment2 import SearchStats, TreeNode, alpha_beta, astar_path, manhattan, minimax


def minimax_value(node: TreeNode, maximizing: bool = True) -> float:
    if not node.children:
        if node.value is None:
            raise ValueError("叶节点缺少 value")
        return node.value
    values = [minimax_value(child, not maximizing) for child in node.children]
    return max(values) if maximizing else min(values)


def clone_ordered(node: TreeNode, maximizing: bool, order: str) -> TreeNode:
    """按子树精确 Minimax 值排序，构造最好序或最差序对照。"""
    if not node.children:
        return TreeNode(value=node.value)
    children = [clone_ordered(child, not maximizing, order) for child in node.children]
    reverse = order == "best"
    children.sort(key=minimax_value, reverse=reverse if maximizing else not reverse)
    return TreeNode(children=children)


def make_tree(seed: int = 20261109, branching: int = 3, depth: int = 5) -> TreeNode:
    rng = random.Random(seed)

    def build(level: int) -> TreeNode:
        if level == depth:
            return TreeNode(value=rng.randint(-50, 50))
        return TreeNode(children=[build(level + 1) for _ in range(branching)])

    return build(0)


def alpha_beta_order_experiment() -> list[dict[str, object]]:
    base = make_tree()
    cases = [
        ("原始顺序", base),
        ("较好顺序", clone_ordered(base, True, "best")),
        ("较差顺序", clone_ordered(base, True, "worst")),
    ]
    rows = []
    repetitions = 300
    for order_name, tree in cases:
        expected = minimax(tree)
        stats = SearchStats()
        began = perf_counter()
        for _ in range(repetitions):
            run_stats = SearchStats()
            value = alpha_beta(tree, stats=run_stats)
            if value != expected:
                raise AssertionError("Alpha-Beta 与 Minimax 结果不一致")
            stats.expanded += run_stats.expanded
            stats.leaves += run_stats.leaves
            stats.pruned += run_stats.pruned
        elapsed_ms = (perf_counter() - began) * 1000 / repetitions
        rows.append({
            "实验": "Alpha-Beta节点顺序", "顺序": order_name,
            "最优值": expected, "节点数": stats.expanded // repetitions,
            "叶节点数": stats.leaves // repetitions,
            "剪枝数": stats.pruned // repetitions,
            "平均耗时_ms": round(elapsed_ms, 4),
        })
    baseline = minimax(base)
    base_stats = SearchStats()
    minimax(base, stats=base_stats)
    rows.insert(0, {
        "实验": "Alpha-Beta节点顺序", "顺序": "Minimax完整搜索",
        "最优值": baseline, "节点数": base_stats.expanded,
        "叶节点数": base_stats.leaves, "剪枝数": 0,
        "平均耗时_ms": "不适用",
    })
    return rows


def make_connected_maze(seed: int, size: int = 15,
                        wall_probability: float = 0.24) -> list[list[str]]:
    rng = random.Random(seed)
    while True:
        grid = [["X" if rng.random() < wall_probability else "." for _ in range(size)]
                for _ in range(size)]
        grid[0][0], grid[-1][-1] = "S", "E"
        path, _, _ = astar_path(grid, (0, 0), (size - 1, size - 1),
                                heuristic=lambda _a, _b: 0)
        if path:
            return grid


def maze_experiment() -> tuple[list[dict[str, object]], list[list[str]]]:
    rows = []
    sample = None
    for seed in range(20261109, 20261129):
        grid = make_connected_maze(seed)
        if sample is None:
            sample = grid
        for label, heuristic in (("h=0", lambda _a, _b: 0), ("曼哈顿", manhattan)):
            path, cost, stats = astar_path(grid, (0, 0), (14, 14), heuristic=heuristic)
            if not path:
                raise AssertionError(f"连通地图搜索失败，seed={seed}")
            rows.append({
                "seed": seed, "启发函数": label, "路径长度": int(cost),
                "扩展节点数": stats.expanded, "耗时_ms": round(stats.elapsed_ms, 4),
            })
    return rows, sample or []


def main() -> None:
    alpha_rows = alpha_beta_order_experiment()
    maze_rows, sample = maze_experiment()
    output = StringIO()
    with redirect_stdout(output):
        print("实验二选做拓展")
        print("\n一、Alpha-Beta 子节点访问顺序")
        print("固定随机种子 20261109，三叉树深度 5；最好/最差顺序按子树 Minimax 值排序。")
        for row in alpha_rows:
            print(row)
        print("\n二、自设迷宫 A* 对照")
        print("地图大小 15x15，障碍概率 0.24，20 个固定种子 20261109 至 20261128。")
        print("样例地图（第一张连通地图）：")
        for line in sample:
            print("".join(line))
        for label in ("h=0", "曼哈顿"):
            subset = [r for r in maze_rows if r["启发函数"] == label]
            print(f"{label}: 平均路径长度={mean(r['路径长度'] for r in subset):.2f}, "
                  f"平均扩展节点={mean(r['扩展节点数'] for r in subset):.2f}, "
                  f"平均耗时={mean(r['耗时_ms'] for r in subset):.4f}ms")
        print("各地图各启发函数的逐次结果另存 CSV。")
    result = output.getvalue()
    print(result, end="")
    folder = Path(__file__).parent
    (folder / "experiment2_extension_results.txt").write_text(result, encoding="utf-8")
    with (folder / "experiment2_extension_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(maze_rows[0]))
        writer.writeheader()
        writer.writerows(maze_rows)
    with (folder / "experiment2_alpha_beta_order.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(alpha_rows[0]))
        writer.writeheader()
        writer.writerows(alpha_rows)
    print("\n结果已保存：experiment2_extension_results.txt、experiment2_extension_results.csv、experiment2_alpha_beta_order.csv")


if __name__ == "__main__":
    main()
