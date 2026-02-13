import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
import ubongosolve as ub

# ===================================================================
# Config
# ===================================================================
MAX_BOARD_CELLS = 100       # max cells in the board
MAX_PIECES = 12             # max number of pieces
MAX_PIECE_CELLS = 12        # max cells per piece
MAX_COORD_VALUE = 20        # coordinate range: 0..20
SOLVE_TIMEOUT = 30          # OR-Tools timeout in seconds

# Rate limiting: per IP
RATE_LIMIT_WINDOW = 60      # seconds
RATE_LIMIT_MAX_REQUESTS = 20  # max /solve requests per window


# ===================================================================
# Rate limiter (simple in-memory, resets on restart)
# ===================================================================
class RateLimiter:
    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        # Prune old entries
        self.requests[key] = [
            t for t in self.requests[key]
            if now - t < RATE_LIMIT_WINDOW
        ]
        if len(self.requests[key]) >= RATE_LIMIT_MAX_REQUESTS:
            return False
        self.requests[key].append(now)
        return True

rate_limiter = RateLimiter()


# ===================================================================
# Security headers middleware
# ===================================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline'"
        )
        return response


# ===================================================================
# App
# ===================================================================
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SecurityHeadersMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Predefined Ubongo pieces
PREDEFINED_PIECES = {}
for name in sorted(x for x in dir(ub.ubongo.pieces) if not x.startswith("_")):
    piece = getattr(ub.ubongo.pieces, name)
    coords = sorted(piece.coordinates)
    PREDEFINED_PIECES[name] = coords


class SolveRequest(BaseModel):
    board: list[list[int]]        # list of [row, col]
    pieces: list[list[list[int]]]  # list of pieces, each piece = list of [row, col]

    @field_validator("board")
    @classmethod
    def validate_board(cls, v):
        if len(v) > MAX_BOARD_CELLS:
            raise ValueError(f"Board exceeds {MAX_BOARD_CELLS} cells")
        for cell in v:
            if len(cell) != 2:
                raise ValueError("Each board cell must be [row, col]")
            if not all(0 <= x <= MAX_COORD_VALUE for x in cell):
                raise ValueError(f"Coordinates must be 0..{MAX_COORD_VALUE}")
        return v

    @field_validator("pieces")
    @classmethod
    def validate_pieces(cls, v):
        if len(v) > MAX_PIECES:
            raise ValueError(f"Too many pieces (max {MAX_PIECES})")
        for piece in v:
            if len(piece) > MAX_PIECE_CELLS:
                raise ValueError(f"Piece exceeds {MAX_PIECE_CELLS} cells")
            if len(piece) == 0:
                raise ValueError("Piece cannot be empty")
            for cell in piece:
                if len(cell) != 2:
                    raise ValueError("Each piece cell must be [row, col]")
                if not all(0 <= x <= MAX_COORD_VALUE for x in cell):
                    raise ValueError(
                        f"Coordinates must be 0..{MAX_COORD_VALUE}"
                    )
        return v


PIECE_COLORS = [
    "#4CAF50", "#2196F3", "#FF9800", "#E91E63",
    "#9C27B0", "#00BCD4", "#FF5722", "#8BC34A",
    "#3F51B5", "#FFEB3B", "#795548", "#607D8B",
]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "predefined_pieces": PREDEFINED_PIECES,
    })


@app.post("/solve", response_class=HTMLResponse)
async def solve(request: Request, data: SolveRequest):
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": "Too many requests. Please wait a minute and try again.",
        })

    # Validate inputs
    if not data.board:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": "Board is empty. Click cells to define a board shape.",
        })
    if not data.pieces:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": "No pieces selected. Add at least one piece.",
        })

    board_coords = set(tuple(c) for c in data.board)
    pieces = [ub.Piece(set(tuple(c) for c in p)) for p in data.pieces]

    # Check: total piece cells == board cells
    total_piece_cells = sum(len(p.coordinates) for p in pieces)
    board_cells = len(board_coords)
    if total_piece_cells != board_cells:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": (
                f"Mismatch: pieces have {total_piece_cells} cells "
                f"but board has {board_cells} cells."
            ),
        })

    board = ub.Board(board_coords)
    puzzle = ub.UbongoPuzzle(pieces=pieces, board=board)
    status = puzzle.solve(timeout=SOLVE_TIMEOUT)

    if status != "OPTIMAL":
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"No solution found (status: {puzzle.status_name}).",
        })

    # Build the result grid
    solution = puzzle.solution  # {(row,col): piece_index}
    all_coords = board_coords
    min_r = min(r for r, c in all_coords)
    max_r = max(r for r, c in all_coords)
    min_c = min(c for r, c in all_coords)
    max_c = max(c for r, c in all_coords)

    grid = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            if (r, c) in solution:
                pidx = solution[(r, c)]
                row.append({
                    "in_board": True,
                    "color": PIECE_COLORS[pidx % len(PIECE_COLORS)],
                    "piece_index": pidx,
                })
            else:
                row.append({"in_board": False})
        grid.append(row)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "grid": grid,
        "error": None,
    })
