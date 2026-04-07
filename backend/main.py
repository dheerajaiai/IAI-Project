from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Tuple
import heapq
import numpy as np
import time

Algorithm = Literal["astar", "bfs", "dfs", "greedy"]
Heuristic = Literal["manhattan", "euclidean", "diagonal"]

app = FastAPI()

# Keep local dev origins enabled so the browser can call the API in class demos.
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
    heuristic: Heuristic


class SolveResponse(BaseModel):
    path: List[Tuple[int, int]]
    explored: List[Tuple[int, int]]
    trace: List[dict]
    stats: dict


def heuristic(a: Tuple[int, int], b: Tuple[int, int], mode: Heuristic) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if mode == "manhattan":
        return dx + dy
    if mode == "euclidean":
        return float((dx ** 2 + dy ** 2) ** 0.5)
    # diagonal / Chebyshev
    return float(max(dx, dy))


def neighbors(pos: Tuple[int, int], rows: int, cols: int):
    r, c = pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def solve_astar(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic):
    rows, cols = len(grid), len(grid[0])
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    explored = []
    visited = set()
    trace = []
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
                "g": round(float(current_g), 3),
                "h": round(float(h), 3),
                "f": round(float(current_g + h), 3),
                "open_size": len(open_heap),
                "closed_size": len(visited),
            }
        )
        step += 1

        if current == end:
            break

        for nr, nc in neighbors(current, rows, cols):
            if grid[nr][nc] == 0:
                continue
            next_node = (nr, nc)
            elevation_delta = abs(grid[nr][nc] - grid[current[0]][current[1]])
            tentative_g = current_g + 1 + elevation_delta

            if tentative_g < g_score.get(next_node, float("inf")):
                g_score[next_node] = tentative_g
                came_from[next_node] = current
                f_score = tentative_g + heuristic(next_node, end, mode)
                heapq.heappush(open_heap, (f_score, tentative_g, next_node))

    return reconstruct_path(came_from, start, end), explored, trace


def solve_greedy(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic):
    rows, cols = len(grid), len(grid[0])
    heap = []
    heapq.heappush(heap, (0.0, start))
    came_from = {}
    explored = []
    visited = set()
    trace = []
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
                "h": round(float(h), 3),
                "open_size": len(heap),
                "closed_size": len(visited),
            }
        )
        step += 1

        if current == end:
            break

        for nr, nc in neighbors(current, rows, cols):
            if grid[nr][nc] == 0:
                continue
            next_node = (nr, nc)
            if next_node in visited:
                continue
            h = heuristic(next_node, end, mode)
            heapq.heappush(heap, (h, next_node))
            if next_node not in came_from:
                came_from[next_node] = current

    return reconstruct_path(came_from, start, end), explored, trace


def solve_bfs(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]):
    rows, cols = len(grid), len(grid[0])
    queue = [start]
    came_from = {}
    explored = []
    visited = {start}
    trace = []
    step = 0

    while queue:
        current = queue.pop(0)
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

        for nr, nc in neighbors(current, rows, cols):
            if grid[nr][nc] == 0:
                continue
            next_node = (nr, nc)
            if next_node in visited:
                continue
            visited.add(next_node)
            came_from[next_node] = current
            queue.append(next_node)

    return reconstruct_path(came_from, start, end), explored, trace


def solve_dfs(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]):
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    came_from = {}
    explored = []
    visited = {start}
    trace = []
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

        for nr, nc in neighbors(current, rows, cols):
            if grid[nr][nc] == 0:
                continue
            next_node = (nr, nc)
            if next_node in visited:
                continue
            visited.add(next_node)
            came_from[next_node] = current
            stack.append(next_node)

    return reconstruct_path(came_from, start, end), explored, trace


def reconstruct_path(came_from, start: Tuple[int, int], end: Tuple[int, int]):
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


@app.post("/solve", response_model=SolveResponse)
def solve(payload: SolveRequest):
    grid = payload.grid
    start = tuple(payload.start)
    end = tuple(payload.end)

    start_time = time.perf_counter()

    if payload.algorithm == "astar":
        path, explored, trace = solve_astar(grid, start, end, payload.heuristic)
    elif payload.algorithm == "greedy":
        path, explored, trace = solve_greedy(grid, start, end, payload.heuristic)
    elif payload.algorithm == "bfs":
        path, explored, trace = solve_bfs(grid, start, end)
    else:
        path, explored, trace = solve_dfs(grid, start, end)

    solve_time_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "path": path,
        "explored": explored,
        "trace": trace,
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

    # Smooth the noise to make it Perlin-like
    for _ in range(4):
        grid = (
            grid
            + np.roll(grid, 1, axis=0)
            + np.roll(grid, -1, axis=0)
            + np.roll(grid, 1, axis=1)
            + np.roll(grid, -1, axis=1)
        ) / 5.0

    # Normalize to 0-1
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid[grid < 0.15] = 0.0

    return {"grid": grid.tolist(), "seed": seed}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
