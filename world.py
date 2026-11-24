"""Toy campus map: '#' = building (wall), '.' = walkable cell."""

DEFAULT_MAP = [
    "......",
    ".##.#.",
    ".##...",
    "...#..",
    ".#...#",
    "......",
]


class World:
    def __init__(self, grid=None):
        self.grid = list(grid or DEFAULT_MAP)
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        self.free = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if self.grid[r][c] == "."]
        self._free = set(self.free)
        self._nb = {(r, c): [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                             if (r + dr, c + dc) in self._free]
                    for (r, c) in self.free}

    def is_free(self, cell):
        return tuple(cell) in self._free

    def neighbors(self, cell):
        return self._nb[tuple(cell)]


WORLD = World()
