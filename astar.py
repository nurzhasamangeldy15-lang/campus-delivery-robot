"""Search: A* (and BFS / UCS for comparison) on the campus grid."""
import heapq
from collections import deque


def _check(world, start, goal):
    for cell in (start, goal):
        if not world.is_free(cell):
            raise ValueError(f"{tuple(cell)} is not a walkable cell")
    return tuple(start), tuple(goal)


def _rebuild(came, cur):
    path = []
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    return path[::-1]


def a_star(world, start, goal, use_heuristic=True):
    """f(n) = g(n) + h(n); g = steps so far, h = Manhattan distance (0 -> UCS/Dijkstra).
    Returns (path or None, number of expanded nodes)."""
    start, goal = _check(world, start, goal)
    h = (lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])) if use_heuristic else (lambda c: 0)
    heap, came, g_cost, expanded = [(h(start), 0, start)], {start: None}, {start: 0}, 0
    while heap:
        _, g, cur = heapq.heappop(heap)
        if g > g_cost[cur]:
            continue
        expanded += 1
        if cur == goal:
            return _rebuild(came, cur), expanded
        for nb in world.neighbors(cur):
            ng = g + 1
            if ng < g_cost.get(nb, 10 ** 9):
                g_cost[nb], came[nb] = ng, cur
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return None, expanded


def bfs(world, start, goal):
    start, goal = _check(world, start, goal)
    queue, came, expanded = deque([start]), {start: None}, 0
    while queue:
        cur = queue.popleft()
        expanded += 1
        if cur == goal:
            return _rebuild(came, cur), expanded
        for nb in world.neighbors(cur):
            if nb not in came:
                came[nb] = cur
                queue.append(nb)
    return None, expanded
