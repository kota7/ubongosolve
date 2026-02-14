#%%
from __future__ import annotations
import itertools
import json
import warnings
from typing import TypeAlias, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from ortools.sat.python import cp_model

Coord: TypeAlias = tuple[int, int]
Coords: TypeAlias = set[Coord]

def _normalize_coordinates(coordinates: Coord) -> Coord:
    # Make sure that all coodinates are non-negative, and
    # the smallest is zero for both x and y axis.
    # Note that this does not ensure that (0, 0) is in the coordinates.
    # E.g., shapes like "_|"
    # Return the new coordinates
    min_x = min(c[0] for c in coordinates)
    min_y = min(c[1] for c in coordinates)
    out = set()
    for x, y in coordinates:
        out.add((x - min_x, y - min_y))
    return out


def _coordindates_as_str(coordinates: Coords, char_positive: str="# ", char_negative: str="  ") -> str:
    coordinates_normalized = _normalize_coordinates(coordinates)
    #print(coordinates_normalized)
    max_x = max(c[0] for c in coordinates_normalized)
    max_y = max(c[1] for c in coordinates_normalized)
 
    flag = []
    for _ in range(max_y + 1):
        flag.append([False] * (max_x + 1))
    #print(flag)
    for x, y in coordinates_normalized:
        flag[y][x] = True
    
    out = "\n".join("".join(char_positive if c else char_negative for c in row) for row in flag)
    return out


def _rotate(coordinates: Coords, angle: int) -> Coords:
    # Returns new coordinates such that
    # the original is rotated counter-clockwise around the origin.
    # Angle must be multiple of 90
    angle = angle % 360
    if angle % 90 != 0:
        raise ValueError(f"rotation angle must be multiple of 90, but received: {angle}")
    if angle == 90:
        return set((y, -x) for x, y in coordinates)
    elif angle == 180:
        return set((-x, -y) for x, y in coordinates)
    elif angle == 270:
        return set((-y, x) for x, y in coordinates)
    else:
        return set(coordinates)


def _flip(coordinates: Coords) -> Coords:
    # Returns new coordinates such that
    # the original is flipped horizontally (over y-axis)
    return set((-x, y) for x, y in coordinates)


class Piece:
    def __init__(self, coordinates: Coords):
        self.coordinates = _normalize_coordinates(coordinates)

    def __str__(self) -> str:
        return _coordindates_as_str(self.coordinates)
    
    def __eq__(self, another: Piece) -> bool:
        return isinstance(another, Piece) and self.coordinates == another.coordinates

    def rotate(self, angle: int) -> Piece:
        return Piece(_rotate(self.coordinates, angle))

    def flip(self) -> Piece:
        return Piece(_flip(self.coordinates))


class Board:
    def __init__(self, coordinates: Coords):
        self.coordinates = _normalize_coordinates(coordinates)

    def __str__(self) -> str:
        return _coordindates_as_str(self.coordinates)


class UbongoPuzzle:
    def __init__(self, pieces: list[Piece], board: Board):
        self.pieces = pieces
        self.board = board
        self.model, self.pieces_ext, self.origin_flags = make_ubongo_problem(pieces, board)
        self.solver = cp_model.CpSolver()
        self.status = 0  # unknown

    def solve(self, timeout: int=100) -> str:
        self.solver.parameters.max_time_in_seconds = timeout
        self.status = self.solver.Solve(self.model)
        return self.solver.StatusName(self.status)

    def findall(self, filepath: str, max_solutions: Optional[int]=None, timeout: int=3600) -> int:
        pieces_ext = self.pieces_ext
        origin_flags = self.origin_flags

        class _Collector(cp_model.CpSolverSolutionCallback):
            def __init__(_self):
                cp_model.CpSolverSolutionCallback.__init__(_self)
                _self._count = 0
                _self._file = open(filepath, "w")

            def on_solution_callback(_self):
                _self._count += 1
                print(f"\rSolution count: {_self._count}", end=" ")
                solution = parse_ubongo_solution(_self, self.pieces_ext, self.origin_flags)
                # To save as json, <convert (x, y) -> id> into <list of [x, y, id]>
                row = [[x, y, id] for (x, y), id in solution.items()]
                json.dump(row, _self._file)
                _self._file.write("\n")

                if max_solutions is not None and _self._count >= max_solutions:
                    print("")
                    print(f"Reached the maximum number of solutions {max_solutions}")
                    _self.StopSearch()

            def close_file(_self):
                _self._file.close()

        self.solver.parameters.max_time_in_seconds = timeout
        self.solver.parameters.enumerate_all_solutions = True

        collector = _Collector()
        status = self.solver.Solve(self.model, collector)
        print()
        collector.close_file()
        return collector._count

    @property
    def status_name(self) -> str:
        return self.solver.StatusName(self.status)

    @property
    def solution(self) -> dict[tuple[int, int], int]:
        if self.status_name not in ("OPTIMAL", "FEASIBLE"):
            raise ValueError("Problem has not been solved yet")
        
        return parse_ubongo_solution(self.solver, self.pieces_ext, self.origin_flags)
    
    def print_solution(self, **kwargs):
        return print_ubongo_solution(self.solution, **kwargs)

    def plot_solution(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        return plot_ubongo_solution(self.solution, **kwargs)


def make_ubongo_problem(pieces: list[Piece], board: Board):
    # Generate a or-tools model that solves the "ubongo" problem
    # i.e. put pieces on the board to fill without overwraps

    model = cp_model.CpModel()

    # Make sure the areas of pieces and board are equal
    piece_area = sum(len(p.coordinates) for p in pieces)
    board_area = len(board.coordinates)
    if piece_area != board_area:
        raise ValueError(f"Piece area != board area ({piece_area} vs {board_area})")

    # Expand the list of pieces by flip and rotation
    pieces_ext = {}
    for id, piece in enumerate(pieces):
        tmp = {}
        for flip, angle in itertools.product([0, 1], [0, 90, 180, 270]):
            p = piece.rotate(angle) if flip == 0 else piece.flip().rotate(angle)
            key = (id, flip, angle)
            if p not in tmp.values():
                tmp[key] = p
            # Here, we skip the orientation if the same shape is already covered
            # This would enhance the computation
            # For example, bar shape "---" would have the exact same shape by flipping or 180 rotation
            # It would have only two unique shapes, instead of eight
        pieces_ext.update(tmp)

    # For each (extended) piece, precompute the candidate origin positions
    origin_candidates = {}
    max_x = max(x for x, _ in board.coordinates)
    max_y = max(y for _, y in board.coordinates)
    for key, piece in pieces_ext.items():
        candidates = set()
        # Loop over all origin candidates.
        # Due to the normalization, all piece contains at least one cell with x=0 and
        # at least one cell with y=0.
        # As a result, the origin coordinates must be within the range of the board;
        # It if is, cell (0, y) or (x, 0) would be off-board.
        # Note that the board is also normalized to have min_x = min_y = 0
        for x, y in itertools.product(range(max_x + 1), range(max_y + 1)):
            # Can we put the origin of the piece at (x, y)?
            if all((x + a, y + b) in board.coordinates for a, b in piece.coordinates):
                candidates.add((x, y))
        
        if len(candidates) > 0:
            origin_candidates[key] = candidates

    # filter pieces_ext as per origin_candidates, so we omit orientation
    # that has no position to locate on the board
    pieces_ext = {k: v for k, v in pieces_ext.items() if k in origin_candidates}

    # Detect edge case: if some piece has no orientation, then the problem is not solvable
    covered_ids = set(id for (id, _, _) in origin_candidates)
    for id, piece in enumerate(pieces):
        if id not in covered_ids:
            raise ValueError(f"Piece {id} cannot be placed anywhere on the board")

    # Flags that indicate the location of the piece origin
    origin_flags = {
        (id, flip, angle, x, y): model.NewBoolVar(f"origin_{id}_{flip}_{angle}_{x}_{y}")
        for (id, flip, angle) in pieces_ext
        for x, y in origin_candidates[id, flip, angle]
    }
    # For each piece id, we must choose exactly one (orientation, origin) pair
    for id in range(len(pieces)):
        flags = [v for (i, _, _, _, _), v in origin_flags.items() if id == i]
        model.AddExactlyOne(flags)

    # Pieces cannot overwrap and board must be fully filled
    # To implement this, we first collect, for each cell on the board, 
    # all "origin_flags" which would cover the cell when positive.
    # Then, exactly one of the flags must be positive
    # for the cell to be filled with no overwrap 
    cell_coverers = {(x, y): [] for x, y in board.coordinates}
    for (id, flip, angle, x, y), flag in origin_flags.items():
        for a, b in pieces_ext[id, flip, angle].coordinates:
            cell_coverers[x+a, y+b].append(flag)
    for flags in cell_coverers.values():
        model.AddExactlyOne(flags)

    return model, pieces_ext, origin_flags


def parse_ubongo_solution(
    solver: cp_model.CpSolver,
    pieces_ext: dict[tuple[int, int, int], Piece],
    origin_flags: dict[tuple[int, int, int, int, int], cp_model.IntVar]
) -> dict[tuple[int, int], int]:
    out = {}
    for (id, flip, angle, x, y), flag in origin_flags.items():
        value = solver.Value(flag)
        if value == 0:
            continue
        piece = pieces_ext[id, flip, angle]
        for a, b in piece.coordinates:
            out[x+a, y+b] = id
    return out


def print_ubongo_solution(
    solution: dict[tuple[int, int], int],
    chars: str="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*#@%=",
    display: bool=True
) -> str:
    max_x = max(x for (x, _) in solution)
    max_y = max(y for (_, y) in solution)

    # Warning when id is out of prepared characters
    maxid = max(solution.values())
    if maxid > len(chars) - 1:
        warnings.warn("We have more pieces than the predefined characters, so the same characters are used repeatedly")
        while maxid > len(chars) - 1:
            chars += chars

    board = []
    for _ in range(max_y + 1):
        board.append([-1] * (max_x + 1))
    for (x, y), id in solution.items():
        board[y][x] = id
    
    out = "\n".join("".join(chars[c] + " " if c >= 0 else "  " for c in row) for row in board)
    if display:
        print(out)
    return out


def plot_ubongo_solution(
    solution: dict[tuple[int, int], int],
    ax=None,
    cmap_name: str = "tab20",
    figsize: tuple[float, float]=(8, 4),
    add_text: bool = True,
    display: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    max_x = max(x for (x, _) in solution)
    max_y = max(y for (_, y) in solution)
    piece_ids = sorted(set(solution.values()))
    n_pieces = len(piece_ids)

    cmap = plt.get_cmap(cmap_name, max(n_pieces, 3))
    color_map = {pid: cmap(i) for i, pid in enumerate(piece_ids)}

    # Draw filled cells
    for (x, y), pid in solution.items():
        color = color_map[pid]
        rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)
        if add_text:
            ax.text(x + 0.5, y + 0.5, str(pid), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="black")

    # Draw thick borders between different pieces
    for (x, y), pid in solution.items():
        # Check each edge: right, top, left, bottom
        for dx, dy, x0, y0, x1, y1 in [
            (1, 0, x+1, y, x+1, y+1),   # right
            (0, 1, x, y+1, x+1, y+1),   # top
            (-1, 0, x, y, x, y+1),       # left
            (0, -1, x, y, x+1, y),       # bottom
        ]:
            neighbor = (x + dx, y + dy)
            if neighbor not in solution or solution[neighbor] != pid:
                ax.plot([x0, x1], [y0, y1], color="black", linewidth=2)

    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(0, max_y + 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    fig.tight_layout()
    if not display:
        plt.close(fig)  # Quick workaround not to display the graph

    return fig, ax


def _hash_ubongo_solution(solution: dict[tuple[int, int], int]) -> int:
    # Compute the original hash value for a ubongo solution
    # such that equivalent ones should have the same hash value.
    # To this end, we collect all equivalent solutions and calculate the minimum
    # Flatten all integers into a long tuple, and calculate the hash
    def _convert_solution(solution, flip, angle):
        keys = []
        values = []
        for key, value in solution.items():
            keys.append(key)
            values.append(value)
        
        if flip == 1:
            keys = [(-x, y) for x, y in keys]
        if angle == 90:
            keys = [(y, -x) for x, y in keys]
        elif angle == 180:
            keys = [(-x, -y) for x, y in keys]
        elif angle == 270:
            keys = [(-y, x) for x, y in keys]
        min_x = min(x for x, _ in keys)
        min_y = min(y for _, y in keys)
        keys = [(x - min_x, y - min_y) for x, y in keys]
        return {key: value for key, value in zip(keys, values)}

    equivalents = []
    for flip, angle in itertools.product([0, 1], [0, 90, 180, 270]):
        equivalents.append(_convert_solution(solution, flip, angle))
    # To make sure cells are ordered, convert them to a list of (x, y, id)
    # Since the cells are order-independent, sort them by (x, y) for the uniqueness
    equivalents = [sorted([(x, y, id) for (x, y), id in e.items()]) for e in equivalents]
    # We define the minimum as the representative one
    rep = min(equivalents)
    # Convert the list to tuple to calculate the hash value
    return hash(tuple(rep))


class ubongo:
    class pieces:
        I2 = Piece([(0,0), (1,0)])

        I3 = Piece([(0,0), (1,0), (2,0)])
        L3 = Piece([(0,0), (1,0), (1,1)])

        I4 = Piece([(0,0), (1,0), (2,0), (3,0)])
        O4 = Piece([(0,0), (1,0), (0,1), (1,1)])
        T4 = Piece([(0,0), (1,0), (2,0), (1,1)])
        S4 = Piece([(1,0), (2,0), (0,1), (1,1)])
        L4 = Piece([(0,0), (1,0), (2,0), (2,1)])

        F5 = Piece([(0,0), (1,0), (2,0), (0,1), (1,1)])
        L5 = Piece([(0,0), (1,0), (2,0), (3,0), (3,1)])
        T5 = Piece([(0,0), (1,0), (2,0), (3,0), (1,1)])
        Z5 = Piece([(0,0), (0,1), (1,1), (2,1), (2,2)])
        
        _all = [I2, I3, L3, I4, O4, T4, S4, L4, F5, L5, T5, Z5]

    sample_problems = [
        {
            "pieces": [pieces.T4, pieces.T5, pieces.S4],
            "board": Board([
                (1,0), (2,0),
                (0,1), (1,1), (2,1), (3,1),
                (1,2), (2,2), (3,2), (4,2),
                (1,3), (2,3), (3,3)
            ])
        },
        {
            "pieces": [pieces.L3, pieces.I3, pieces.T5, pieces.L5],
            "board": Board([
                (1,0), (2,0),
                (1,1), (2,1), (3,1),
                (1,2), (2,2), (3,2),
                (0,3), (1,3), (2,3), (3,3),
                (0,4), (1,4),
                (0,5), (1,5)
            ])
        },
    ]


class whitechocolate:
    # White Chocolate Puzzle
    # https://store.hanayamatoys.co.jp/items/58611532
    # https://www.amazon.co.jp/dp/B0BGSH79VJ
    pieces = [
        Piece([[0,0], [0,1], [0,2], [0,3], [0,4], [1,4]]),
        Piece([[0,0], [0,1], [0,2], [0,3], [1,3], [2,3]]),
        Piece([[0,0], [0,1], [0,2], [1,1], [1,2]]),
        Piece([[0,0], [0,1], [0,2], [0,3], [1,3]]),
        Piece([[0,0], [0,1], [0,2], [1,2]]),
        Piece([[0,0], [0,1], [0,2], [0,3], [1,2], [1,3]]),
        Piece([[0,0], [0,1], [1,1]]),
        Piece([[0,0], [0,1], [0,2], [1,2], [2,2]]),
    ]

    board = Board([(x, y) for x, y in itertools.product(range(8), range(5))])

    boards = {
        # Basic board of 5 x 8 rectangle
        "basic": Board([(x, y) for x, y in itertools.product(range(8), range(5))])

        # Split boards
        ,"split_1": Board([
            [0,0], [1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0],
            [0,1], [1,1], [2,1], [3,1], [4,1], [5,1], [6,2], [7,2],
            [0,2], [1,2], [2,2], [3,2], [4,3], [5,3], [6,3], [7,3],
            [0,3], [1,3], [2,4], [3,4], [4,4], [5,4], [6,4], [7,4],
            [0,5], [1,5], [2,5], [3,5], [4,5], [5,5], [6,5], [7,5],
        ])
        ,"split_2": Board([
            [0,0], [1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [8,1],
            [0,1], [1,1], [2,1], [3,1], [5,2], [6,2], [7,2], [8,2],
            [0,2], [1,2], [2,2], [3,2], [5,3], [6,3], [7,3], [8,3],
            [0,3], [1,3], [2,3], [3,3], [5,4], [6,4], [7,4], [8,4],
            [0,4], [2,5], [3,5], [4,5], [5,5], [6,5], [7,5], [8,5],
        ])

        # Challenge boards
        ,"challenge_1": Board([
            [0,0], [1,0], [2,0], [3,0],
            [0,1], [1,1], [2,1], [3,1],
            [0,2], [1,2], [2,2], [3,2],
            [0,3], [1,3], [2,3], [3,3], [4,3], [5,3], [6,3],
            [0,4], [1,4], [2,4], [3,4], [4,4], [5,4], [6,4],
            [0,5], [1,5], [2,5], [3,5], [4,5], [5,5], [6,5],
            [0,6], [1,6], [2,6], [3,6], [4,6], [5,6], [6,6],
        ])
        ,"challenge_2": Board([
            [3,0],
            [1,1], [2,1], [3,1], [4,1], [5,1], [6,1],
            [1,2], [2,2], [3,2], [4,2], [5,2], [6,2],
            [1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3],
            [0,4], [1,4], [2,4], [3,4], [4,4], [5,4], [6,4],
            [1,5], [2,5], [3,5], [4,5], [5,5], [6,5],
            [1,6], [2,6], [3,6], [4,6], [5,6], [6,6],
            [4,7]
        ])
        ,"challenge_3": Board([
            [4,0], [5,0],
            [4,1], [5,1], [6,1], [7,1], 
            [0,2], [1,2], [2,2], [3,2], [6,2], [7,2],
            [0,3], [1,3], [2,3], [3,3], 
            [0,4], [1,4], [2,4], [3,4], [4,4], [5,4], [6,4], [7,4], 
            [0,5], [1,5], [2,5], [3,5], [4,5], [5,5], [6,5], [7,5],
            [4,6], [5,6], [6,6], [7,6],
            [4,7], [5,7], [6,7], [7,7],
        ])
        ,"challenge_4": Board([
            [2,0], [3,0],
            [2,1], [3,1],
            [0,2], [1,2], [2,2], [3,2], [4,2], [5,2], [6,2], [7,2],
            [0,3], [1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3],
            [0,5], [1,5], [2,5], [3,5], [4,5], [5,5], [6,5], [7,5],
            [0,6], [1,6], [2,6], [3,6], [4,6], [5,6], [6,6], [7,6],
            [2,7], [3,7],
            [2,8], [3,8],
        ])
        ,"challenge_5": Board([
            [2,0], [3,0],
            [2,1], [3,1], [4,1], [5,1],
            [1,2], [2,2], [3,2], [4,2], [5,2], [6,2], [7,2],
            [1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3],
            [0,4], [1,4], [2,4], [3,4], [4,4], [5,4], [6,4],
            [0,5], [1,5], [2,5], [3,5], [4,5], [5,5], [6,5],
            [2,6], [3,6], [4,6], [5,6],
            [4,7], [5,7],
        ])
        ,"challenge_6": Board([
            [4,0], [5,0], [6,0], [7,0],
            [3,1], [4,1], [5,1], [6,1], [7,1], [8,1],
            [2,2], [3,2], [4,2], [5,2], [6,2], [7,2], [8,2], [9,2],
            [1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3], [8,3], [9,3], [10,3],
            [0,4], [1,4], [2,4], [3,4], [4,4], [5,4], [6,4], [7,4], [8,4], [9,4], [10,4], [11,4],
        ])
    }

#%%
if __name__ == "__main__":
    #%% Baic test
    # Define pieces as sets of (x, y) coordinates
    pieces = [
        Piece([(0,0), (0,1), (0,2), (0,3), (1,3)]), # L-shape
        Piece([(0,0), (1,0), (1,1), (2,1)]),        # S-shape
        Piece([(0,0), (0,1), (1,1)]),               # mini L-shape
    ]

    # Define a board
    board = Board([(x, y) for x in range(4) for y in range(3)])

    # Solve
    solver = UbongoPuzzle(pieces, board)
    status = solver.solve()
    print(status)  # "OPTIMAL" or "FEASIBLE"

    # View the solution
    solver.print_solution()     # text output
    solver.plot_solution()      # matplotlib figure

    #%% Ubongo sample
    for problem in ubongo.sample_problems:
        p = UbongoPuzzle(problem["pieces"], problem["board"])
        status = p.solve()
        print(status)
        #p.print_solution()
        p.plot_solution()

    #%% White Chocolate example
    p = UbongoPuzzle(whitechocolate.pieces, whitechocolate.board)
    status = p.solve()
    print(status)
    #p.print_solution()
    p.plot_solution()

    #%% White Chocolate example, more
    for name, board in whitechocolate.boards.items():
        p = UbongoPuzzle(whitechocolate.pieces, board)
        status = p.solve()
        print(f"{name}: {status}")
        #p.print_solution()
        p.plot_solution()
        print()

    #%% Find all solutions to white chocolate problem (takes a few minutes)
    p = UbongoPuzzle(whitechocolate.pieces, whitechocolate.board)
    n = p.findall("chocolate_all.jsonl")
    # We found 31056 solutions, but the package says there are 7764 solution.
    # Actually 31056 = 7764 * 4.
    # We have 4x solutions because the solver does not distinguish identical patterns;
    # Four patterns are original, 180-rotated, flip-horizontally, and flip-vertically

    #%% Visualize all chcolate solutions (takes a few minutes)
    import os
    import json
    savedir = "chocolate_solutions"
    os.makedirs(savedir, exist_ok=True)
    with open("chocolate_all.jsonl") as f:
        ct = 0
        finished = set()
        for row in f:
            tmp = json.loads(row.strip())
            solution = {(x, y): id for (x, y, id) in tmp}
            # Check if a duplicate solution is already finished
            hashvalue = _hash_ubongo_solution(solution)
            if hashvalue in finished:
                continue
            ct += 1
            fig, ax = plot_ubongo_solution(solution, display=False, figsize=(3.2, 2))
            savename = os.path.join(savedir, f"sol_{ct}.png")
            fig.savefig(savename)
            fig, ax = plot_ubongo_solution(solution, display=False, figsize=(3.2, 2), add_text=False)
            savename = os.path.join(savedir, f"sol_notext_{ct}.png")
            fig.savefig(savename)

            print(f"\rSaved solution {ct}", end="")
            finished.add(hashvalue)
        print()
        print(f"Finished. Final solution count: {ct}")

    #%% Make a pdf document with all solutions
    from glob import glob
    import os
    import re
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from PIL import Image    
    imagefiles = glob(os.path.join("chocolate_solutions", "*_notext_*.png"))
    # sort by number
    imagefiles.sort(key=lambda f: int(re.search(r"_(\d+)\.png", os.path.basename(f)).group(1)))
    print(f"{len(imagefiles)} images found: {imagefiles[:5]} ...")
    assert len(imagefiles) > 0
    # document settings
    pdffile = "choco_allsols.pdf"
    papersize = A4
    cols = 3
    rows = 7
    margin_x = 0.3 * inch
    margin_y = 0.3 * inch
    # calculate length
    images_per_page = cols * rows
    page_width, page_height = papersize
    available_width = page_width - 2 * margin_x
    available_height = page_height - 2 * margin_y    
    cell_width = available_width / cols
    cell_height = available_height / rows    
    first_img = Image.open(imagefiles[0])
    img_width, img_height = first_img.size
    img_aspect = img_width / img_height
    padding = 0.05 * inch
    max_img_width = cell_width - padding
    max_img_height = cell_height - padding
    # Fit image maintaining aspect ratio
    if max_img_width / img_aspect < max_img_height:
        display_width = max_img_width
        display_height = max_img_width / img_aspect
    else:
        display_height = max_img_height
        display_width = max_img_height * img_aspect
    # start editing the document
    print(f"Start editing {pdffile}")
    c = canvas.Canvas(pdffile, papersize=papersize)
    for idx, img_path in enumerate(imagefiles):
        print(f"\r{idx+1}/{len(imagefiles)} ({100 * (idx+1) // len(imagefiles)}%)", end="")
        
        # Position in grid
        item_on_page = idx % images_per_page
        col = item_on_page % cols
        row = item_on_page // cols

        # Start new page if needed
        if item_on_page == 0 and idx > 0:
            c.showPage()
            # debug
            #break
        
        # Calculate position (origin is bottom-left in reportlab)
        # We need to flip rows since we count from top but reportlab counts from bottom
        x = margin_x + col * cell_width + (cell_width - display_width) / 2
        y = page_height - margin_y - (row + 1) * cell_height + (cell_height - display_height) / 2

        # Draw image
        c.drawImage(
            img_path, x, y, 
            width=display_width, 
            height=display_height,
            preserveAspectRatio=True
        )
        number = re.search(r"_(\d+)\.png", os.path.basename(img_path)).group(1)
        c.setFont("Helvetica", 8)
        text_x = x + 15
        c.drawString(text_x, y + display_height - 4, f"Solution {number}")
    c.save()  # save the last page
    print()
    print("Finished")

# %%
