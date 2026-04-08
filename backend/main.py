from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

Algorithm = Literal[
    "bfs",
    "dfs",
    "dls",
    "iddfs",
    "ucs",
    "bidirectional",
    "greedy",
    "astar",
    "idastar",
    "beam",
]
Heuristic = Literal["manhattan", "euclidean", "diagonal"]

app = FastAPI()

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://pathmind.vercel.app",
    "https://pathmind-classroom-demo-1wt57i6fe-dheerajsai0416-9884s-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    grid: List[List[float]]
    start: Tuple[int, int]
    end: Tuple[int, int]
    algorithm: Algorithm
    heuristic: Heuristic = "euclidean"
    depth_limit: int = 20
    beam_width: int = 6


class SolveResponse(BaseModel):
    path: List[Tuple[int, int]]
    explored: List[Tuple[int, int]]
    trace: List[dict]
    summary: str
    stats: dict


def heuristic(a: Tuple[int, int], b: Tuple[int, int], mode: Heuristic) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if mode == "manhattan":
        return float(dx + dy)
    if mode == "euclidean":
        return float((dx**2 + dy**2) ** 0.5)
    return float(max(dx, dy))


def step_cost(grid: List[List[float]], a: Tuple[int, int], b: Tuple[int, int]) -> float:
    ar, ac = a
    br, bc = b
    diagonal = 1.4 if ar != br and ac != bc else 1.0
    slope = abs(grid[br][bc] - grid[ar][ac])
    return diagonal + slope


def neighbors(pos: Tuple[int, int], rows: int, cols: int):
    r, c = pos
    for dr, dc in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def validate_grid_request(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]
) -> None:
    if not grid or not grid[0]:
        raise HTTPException(status_code=400, detail="Grid cannot be empty")

    rows = len(grid)
    cols = len(grid[0])
    sr, sc = start
    er, ec = end

    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= er < rows and 0 <= ec < cols):
        raise HTTPException(status_code=400, detail="Start or target is out of bounds")

    if grid[sr][sc] == 0 or grid[er][ec] == 0:
        raise HTTPException(
            status_code=400, detail="Start and target nodes must be on passable terrain"
        )


def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    if start == end:
        return [start]
    if end not in came_from:
        return []

    path = [end]
    current = end
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def solve_bfs(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]):
    rows, cols = len(grid), len(grid[0])
    queue = deque([start])
    visited = {start}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    while queue:
        current = queue.popleft()
        explored.append(current)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "frontier_size": len(queue),
                "visited_size": len(visited),
            }
        )
        step += 1

        if current == end:
            break

        for nxt in neighbors(current, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            queue.append(nxt)

    return reconstruct_path(came_from, start, end), explored, trace


def solve_dfs(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]):
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    visited = {start}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    while stack:
        current = stack.pop()
        explored.append(current)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "frontier_size": len(stack),
                "visited_size": len(visited),
            }
        )
        step += 1

        if current == end:
            break

        nxt_list = list(neighbors(current, rows, cols))
        nxt_list.reverse()
        for nxt in nxt_list:
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            stack.append(nxt)

    return reconstruct_path(came_from, start, end), explored, trace


def solve_dls(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], depth_limit: int
):
    rows, cols = len(grid), len(grid[0])
    stack = [(start, 0)]
    best_depth = {start: 0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []
    found = False

    step = 0
    while stack:
        current, depth = stack.pop()
        explored.append(current)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "depth": depth,
                "limit": depth_limit,
                "frontier_size": len(stack),
            }
        )
        step += 1

        if current == end:
            found = True
            break
        if depth >= depth_limit:
            continue

        nxt_list = list(neighbors(current, rows, cols))
        nxt_list.reverse()
        for nxt in nxt_list:
            nr, nc = nxt
            if grid[nr][nc] == 0:
                continue
            nd = depth + 1
            if nd < best_depth.get(nxt, 10**9):
                best_depth[nxt] = nd
                came_from[nxt] = current
                stack.append((nxt, nd))

    return (reconstruct_path(came_from, start, end) if found else []), explored, trace


def solve_iddfs(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], max_depth: int
):
    merged_explored: List[Tuple[int, int]] = []
    merged_trace: List[dict] = []

    for depth in range(max_depth + 1):
        path, explored, trace = solve_dls(grid, start, end, depth)
        merged_explored.extend(explored)
        for item in trace:
            merged_trace.append({**item, "iteration_limit": depth})
        if path:
            return path, merged_explored, merged_trace

    return [], merged_explored, merged_trace


def solve_ucs(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]):
    rows, cols = len(grid), len(grid[0])
    heap = [(0.0, start)]
    dist = {start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = set()
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    while heap:
        current_cost, current = heapq.heappop(heap)
        if current in visited:
            continue

        visited.add(current)
        explored.append(current)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "g": round(current_cost, 3),
                "open_size": len(heap),
                "closed_size": len(visited),
            }
        )
        step += 1

        if current == end:
            break

        for nxt in neighbors(current, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0:
                continue
            candidate = current_cost + step_cost(grid, current, nxt)
            if candidate < dist.get(nxt, float("inf")):
                dist[nxt] = candidate
                came_from[nxt] = current
                heapq.heappush(heap, (candidate, nxt))

    return reconstruct_path(came_from, start, end), explored, trace


def solve_bidirectional(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]
):
    rows, cols = len(grid), len(grid[0])
    queue_f = deque([start])
    queue_b = deque([end])
    parent_f: Dict[Tuple[int, int], Tuple[int, int]] = {}
    parent_b: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited_f = {start}
    visited_b = {end}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    meeting: Optional[Tuple[int, int]] = None
    step = 0
    while queue_f and queue_b:
        current_f = queue_f.popleft()
        explored.append(current_f)
        trace.append(
            {
                "step": step,
                "node": [current_f[0], current_f[1]],
                "wave": "forward",
                "frontier_size": len(queue_f),
            }
        )
        step += 1

        if current_f in visited_b:
            meeting = current_f
            break

        for nxt in neighbors(current_f, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited_f:
                continue
            visited_f.add(nxt)
            parent_f[nxt] = current_f
            queue_f.append(nxt)

        current_b = queue_b.popleft()
        explored.append(current_b)
        trace.append(
            {
                "step": step,
                "node": [current_b[0], current_b[1]],
                "wave": "backward",
                "frontier_size": len(queue_b),
            }
        )
        step += 1

        if current_b in visited_f:
            meeting = current_b
            break

        for nxt in neighbors(current_b, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited_b:
                continue
            visited_b.add(nxt)
            parent_b[nxt] = current_b
            queue_b.append(nxt)

    if meeting is None:
        return [], explored, trace

    left = [meeting]
    node = meeting
    while node != start:
        node = parent_f[node]
        left.append(node)
    left.reverse()

    right: List[Tuple[int, int]] = []
    node = meeting
    while node != end:
        node = parent_b[node]
        right.append(node)

    return left + right, explored, trace


def solve_greedy(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic
):
    rows, cols = len(grid), len(grid[0])
    heap = [(heuristic(start, end, mode), start)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = set()
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    while heap:
        _, current = heapq.heappop(heap)
        if current in visited:
            continue

        visited.add(current)
        explored.append(current)

        h = heuristic(current, end, mode)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "h": round(h, 3),
                "open_size": len(heap),
                "closed_size": len(visited),
            }
        )
        step += 1

        if current == end:
            break

        for nxt in neighbors(current, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited:
                continue
            if nxt not in came_from:
                came_from[nxt] = current
            heapq.heappush(heap, (heuristic(nxt, end, mode), nxt))

    return reconstruct_path(came_from, start, end), explored, trace


def solve_astar(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic
):
    rows, cols = len(grid), len(grid[0])
    open_heap = [(0.0, 0.0, start)]
    g_score = {start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = set()
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue

        visited.add(current)
        explored.append(current)

        h = heuristic(current, end, mode)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "g": round(current_g, 3),
                "h": round(h, 3),
                "f": round(current_g + h, 3),
                "open_size": len(open_heap),
                "closed_size": len(visited),
            }
        )
        step += 1

        if current == end:
            break

        for nxt in neighbors(current, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0:
                continue
            candidate = current_g + step_cost(grid, current, nxt)
            if candidate < g_score.get(nxt, float("inf")):
                g_score[nxt] = candidate
                came_from[nxt] = current
                f = candidate + heuristic(nxt, end, mode)
                heapq.heappush(open_heap, (f, candidate, nxt))

    return reconstruct_path(came_from, start, end), explored, trace


def solve_idastar(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic
):
    rows, cols = len(grid), len(grid[0])
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    def search(path: List[Tuple[int, int]], g: float, bound: float, visited: set):
        node = path[-1]
        f = g + heuristic(node, end, mode)
        if f > bound:
            return f, None

        explored.append(node)
        trace.append(
            {
                "step": len(trace),
                "node": [node[0], node[1]],
                "g": round(g, 3),
                "f_bound": round(bound, 3),
            }
        )

        if node == end:
            return f, list(path)

        minimum = float("inf")
        for nxt in neighbors(node, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            t, result = search(path, g + step_cost(grid, node, nxt), bound, visited)
            if result is not None:
                return t, result
            minimum = min(minimum, t)
            path.pop()
            visited.remove(nxt)

        return minimum, None

    bound = heuristic(start, end, mode)
    path = [start]
    visited = {start}

    while True:
        t, result = search(path, 0.0, bound, visited)
        if result is not None:
            return result, explored, trace
        if t == float("inf"):
            return [], explored, trace
        bound = t


def solve_beam(
    grid: List[List[float]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    mode: Heuristic,
    beam_width: int,
):
    rows, cols = len(grid), len(grid[0])
    layer = [start]
    visited = {start}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    while layer:
        candidates: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []

        for node in layer:
            explored.append(node)
            trace.append(
                {
                    "step": step,
                    "node": [node[0], node[1]],
                    "beam_width": beam_width,
                    "layer_size": len(layer),
                }
            )
            step += 1

            if node == end:
                return reconstruct_path(came_from, start, end), explored, trace

            for nxt in neighbors(node, rows, cols):
                nr, nc = nxt
                if grid[nr][nc] == 0 or nxt in visited:
                    continue
                candidates.append((heuristic(nxt, end, mode), nxt, node))

        candidates.sort(key=lambda x: x[0])
        next_layer: List[Tuple[int, int]] = []
        for _, nxt, parent in candidates[: max(1, beam_width)]:
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = parent
            next_layer.append(nxt)

        layer = next_layer

    return [], explored, trace


@app.post("/solve", response_model=SolveResponse)
def solve(payload: SolveRequest):
    grid = payload.grid
    start = tuple(payload.start)
    end = tuple(payload.end)
    validate_grid_request(grid, start, end)

    start_time = time.perf_counter()

    if payload.algorithm == "bfs":
        path, explored, trace = solve_bfs(grid, start, end)
    elif payload.algorithm == "dfs":
        path, explored, trace = solve_dfs(grid, start, end)
    elif payload.algorithm == "dls":
        path, explored, trace = solve_dls(grid, start, end, payload.depth_limit)
    elif payload.algorithm == "iddfs":
        path, explored, trace = solve_iddfs(grid, start, end, payload.depth_limit)
    elif payload.algorithm == "ucs":
        path, explored, trace = solve_ucs(grid, start, end)
    elif payload.algorithm == "bidirectional":
        path, explored, trace = solve_bidirectional(grid, start, end)
    elif payload.algorithm == "greedy":
        path, explored, trace = solve_greedy(grid, start, end, payload.heuristic)
    elif payload.algorithm == "astar":
        path, explored, trace = solve_astar(grid, start, end, payload.heuristic)
    elif payload.algorithm == "idastar":
        path, explored, trace = solve_idastar(grid, start, end, payload.heuristic)
    else:
        path, explored, trace = solve_beam(
            grid, start, end, payload.heuristic, payload.beam_width
        )

    solve_time_ms = int((time.perf_counter() - start_time) * 1000)
    solved = bool(path)
    summary = (
        f"{payload.algorithm.upper()} {'found a route' if solved else 'did not find a route'} "
        f"from {start} to {end}."
    )

    return {
        "path": path,
        "explored": explored,
        "trace": trace,
        "summary": summary,
        "stats": {
            "nodes_explored": len(explored),
            "path_length": len(path),
            "solve_time_ms": solve_time_ms,
        },
    }


@app.get("/terrain")
def terrain(seed: int = 42):
    rng = np.random.default_rng(seed)
    grid = rng.random((50, 50))

    for _ in range(4):
        grid = (
            grid
            + np.roll(grid, 1, axis=0)
            + np.roll(grid, -1, axis=0)
            + np.roll(grid, 1, axis=1)
            + np.roll(grid, -1, axis=1)
        ) / 5.0

    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid[grid < 0.15] = 0.0

    return {"grid": grid.tolist(), "seed": seed}


@app.get("/algorithms")
def algorithms_catalog():
    return {
        "terrain": [
            "bfs",
            "dfs",
            "dls",
            "iddfs",
            "ucs",
            "bidirectional",
            "greedy",
            "astar",
            "idastar",
            "beam",
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
