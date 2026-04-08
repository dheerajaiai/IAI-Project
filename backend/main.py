from __future__ import annotations

import heapq
import math
import random
import time
from collections import Counter, deque
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GridAlgorithm = Literal[
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
    "hill_simple",
    "hill_steepest",
    "hill_stochastic",
    "simulated_annealing",
    "local_beam",
    "genetic",
    "means_end",
    "ao_star",
]

ScenarioAlgorithm = Literal[
    "minimax",
    "alpha_beta",
    "forward_chaining",
    "backward_chaining",
    "resolution",
    "csp_backtracking",
    "forward_checking",
    "ac3",
    "id3",
    "naive_bayes",
    "knn",
    "bayesian_inference",
    "bayesian_network",
    "hmm",
    "strips",
]

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
    "hill_simple",
    "hill_steepest",
    "hill_stochastic",
    "simulated_annealing",
    "local_beam",
    "genetic",
    "means_end",
    "ao_star",
    "minimax",
    "alpha_beta",
    "forward_chaining",
    "backward_chaining",
    "resolution",
    "csp_backtracking",
    "forward_checking",
    "ac3",
    "id3",
    "naive_bayes",
    "knn",
    "bayesian_inference",
    "bayesian_network",
    "hmm",
    "strips",
]

Heuristic = Literal["manhattan", "euclidean", "diagonal"]

GRID_ALGORITHMS = {
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
    "hill_simple",
    "hill_steepest",
    "hill_stochastic",
    "simulated_annealing",
    "local_beam",
    "genetic",
    "means_end",
    "ao_star",
}

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
    frontier = deque([start])
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    while frontier:
        current = frontier.popleft()
        explored.append(current)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "frontier_size": len(frontier),
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
            frontier.append(nxt)

    return reconstruct_path(came_from, start, end), explored, trace


def solve_dfs(grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]):
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}
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

        next_nodes = list(neighbors(current, rows, cols))
        next_nodes.reverse()
        for nxt in next_nodes:
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
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best_depth = {start: 0}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    step = 0
    found = False
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

        next_nodes = list(neighbors(current, rows, cols))
        next_nodes.reverse()
        for nxt in next_nodes:
            nr, nc = nxt
            if grid[nr][nc] == 0:
                continue
            nd = depth + 1
            if nd < best_depth.get(nxt, 10**9):
                best_depth[nxt] = nd
                came_from[nxt] = current
                stack.append((nxt, nd))

    path = reconstruct_path(came_from, start, end) if found else []
    return path, explored, trace


def solve_iddfs(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], max_depth: int
):
    merged_explored: List[Tuple[int, int]] = []
    merged_trace: List[dict] = []

    for depth in range(max_depth + 1):
        path, explored, trace = solve_dls(grid, start, end, depth)
        merged_explored.extend(explored)
        for t in trace:
            merged_trace.append({**t, "iteration_limit": depth})
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
                "g": round(float(current_cost), 3),
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
            ncst = current_cost + step_cost(grid, current, nxt)
            if ncst < dist.get(nxt, float("inf")):
                dist[nxt] = ncst
                came_from[nxt] = current
                heapq.heappush(heap, (ncst, nxt))

    return reconstruct_path(came_from, start, end), explored, trace


def solve_bidirectional(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int]
):
    rows, cols = len(grid), len(grid[0])
    q_f = deque([start])
    q_b = deque([end])
    parent_f: Dict[Tuple[int, int], Tuple[int, int]] = {}
    parent_b: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited_f = {start}
    visited_b = {end}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    meet: Optional[Tuple[int, int]] = None
    step = 0
    while q_f and q_b:
        current_f = q_f.popleft()
        explored.append(current_f)
        trace.append(
            {
                "step": step,
                "node": [current_f[0], current_f[1]],
                "wave": "forward",
                "frontier_size": len(q_f),
            }
        )
        step += 1
        if current_f in visited_b:
            meet = current_f
            break

        for nxt in neighbors(current_f, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited_f:
                continue
            visited_f.add(nxt)
            parent_f[nxt] = current_f
            q_f.append(nxt)

        current_b = q_b.popleft()
        explored.append(current_b)
        trace.append(
            {
                "step": step,
                "node": [current_b[0], current_b[1]],
                "wave": "backward",
                "frontier_size": len(q_b),
            }
        )
        step += 1
        if current_b in visited_f:
            meet = current_b
            break

        for nxt in neighbors(current_b, rows, cols):
            nr, nc = nxt
            if grid[nr][nc] == 0 or nxt in visited_b:
                continue
            visited_b.add(nxt)
            parent_b[nxt] = current_b
            q_b.append(nxt)

    if not meet:
        return [], explored, trace

    path_start = [meet]
    node = meet
    while node != start:
        node = parent_f[node]
        path_start.append(node)
    path_start.reverse()

    path_end: List[Tuple[int, int]] = []
    node = meet
    while node != end:
        node = parent_b[node]
        path_end.append(node)

    return path_start + path_end, explored, trace


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
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0.0}
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
            tentative = current_g + step_cost(grid, current, nxt)
            if tentative < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative
                came_from[nxt] = current
                f = tentative + heuristic(nxt, end, mode)
                heapq.heappush(open_heap, (f, tentative, nxt))

    return reconstruct_path(came_from, start, end), explored, trace


def solve_idastar(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic
):
    rows, cols = len(grid), len(grid[0])
    trace: List[dict] = []
    explored: List[Tuple[int, int]] = []

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
        next_layer = []

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
                score = heuristic(nxt, end, mode)
                candidates.append((score, nxt, node))

        candidates.sort(key=lambda x: x[0])
        for _, nxt, parent in candidates[: max(1, beam_width)]:
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = parent
            next_layer.append(nxt)

        layer = next_layer

    return [], explored, trace


def pick_best_neighbor(
    grid: List[List[float]],
    current: Tuple[int, int],
    end: Tuple[int, int],
    mode: Heuristic,
    visited: set,
):
    rows, cols = len(grid), len(grid[0])
    options = []
    for nxt in neighbors(current, rows, cols):
        nr, nc = nxt
        if grid[nr][nc] == 0 or nxt in visited:
            continue
        options.append((heuristic(nxt, end, mode), nxt))
    options.sort(key=lambda x: x[0])
    return options


def solve_hill(
    grid: List[List[float]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    mode: Heuristic,
    variant: str,
):
    current = start
    visited = {current}
    path = [current]
    explored = [current]
    trace: List[dict] = []

    max_steps = len(grid) * len(grid[0])
    temperature = 2.5

    for step in range(max_steps):
        h_cur = heuristic(current, end, mode)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "h": round(h_cur, 3),
                "variant": variant,
            }
        )
        if current == end:
            break

        options = pick_best_neighbor(grid, current, end, mode, visited)
        if not options:
            break

        if variant == "simple":
            next_node = options[0][1]
            if options[0][0] >= h_cur:
                break
        elif variant == "steepest":
            best = min(options, key=lambda x: x[0])
            next_node = best[1]
            if best[0] >= h_cur:
                break
        elif variant == "stochastic":
            better = [opt for opt in options if opt[0] < h_cur]
            if not better:
                break
            next_node = random.choice(better[: min(4, len(better))])[1]
        else:  # simulated annealing
            candidate = random.choice(options[: min(6, len(options))])
            delta = candidate[0] - h_cur
            if delta < 0:
                next_node = candidate[1]
            else:
                prob = math.exp(-delta / max(temperature, 1e-6))
                if random.random() > prob:
                    temperature *= 0.95
                    continue
                next_node = candidate[1]
            temperature *= 0.95

        current = next_node
        visited.add(current)
        path.append(current)
        explored.append(current)

    if path[-1] != end:
        return [], explored, trace
    return path, explored, trace


def solve_local_beam(
    grid: List[List[float]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    mode: Heuristic,
    beam_width: int,
):
    rows, cols = len(grid), len(grid[0])
    states = [start]
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}
    explored: List[Tuple[int, int]] = []
    trace: List[dict] = []

    for step in range(rows * cols):
        scored: List[Tuple[float, Tuple[int, int], Optional[Tuple[int, int]]]] = []
        for state in states:
            explored.append(state)
            trace.append(
                {
                    "step": len(trace),
                    "node": [state[0], state[1]],
                    "beam_size": len(states),
                }
            )
            if state == end:
                return reconstruct_path(parents, start, end), explored, trace

            for nxt in neighbors(state, rows, cols):
                nr, nc = nxt
                if grid[nr][nc] == 0 or nxt in visited:
                    continue
                scored.append((heuristic(nxt, end, mode), nxt, state))

        if not scored:
            break

        scored.sort(key=lambda x: x[0])
        next_states = []
        for _, node, parent in scored[: max(1, beam_width)]:
            if node in visited:
                continue
            visited.add(node)
            if parent is not None:
                parents[node] = parent
            next_states.append(node)

        states = next_states
        if not states:
            break

    return [], explored, trace


def solve_genetic(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic
):
    rows, cols = len(grid), len(grid[0])
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    chromosome_len = 70
    pop_size = 40

    def simulate(chrom):
        r, c = start
        path = [start]
        for dr, dc in chrom:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 0:
                r, c = nr, nc
                path.append((r, c))
            if (r, c) == end:
                break
        dist = heuristic((r, c), end, mode)
        fitness = 1.0 / (1.0 + dist)
        return fitness, path

    population = [[random.choice(moves) for _ in range(chromosome_len)] for _ in range(pop_size)]
    trace: List[dict] = []
    best_path = [start]

    for generation in range(40):
        scored = []
        for chrom in population:
            fit, path = simulate(chrom)
            scored.append((fit, chrom, path))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_fit, _, curr_best_path = scored[0]
        best_path = curr_best_path
        trace.append(
            {
                "step": generation,
                "generation": generation,
                "best_fitness": round(best_fit, 5),
                "best_path_length": len(curr_best_path),
            }
        )

        if curr_best_path and curr_best_path[-1] == end:
            explored = list(dict.fromkeys(curr_best_path))
            return curr_best_path, explored, trace

        elite = [scored[i][1] for i in range(8)]
        next_pop = elite[:]
        while len(next_pop) < pop_size:
            a, b = random.sample(elite, 2)
            cut = random.randint(1, chromosome_len - 2)
            child = a[:cut] + b[cut:]
            if random.random() < 0.25:
                m = random.randint(0, chromosome_len - 1)
                child[m] = random.choice(moves)
            next_pop.append(child)
        population = next_pop

    explored = list(dict.fromkeys(best_path))
    return (best_path if best_path and best_path[-1] == end else []), explored, trace


def solve_means_end(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic
):
    current = start
    path = [current]
    explored = [current]
    trace: List[dict] = []
    visited = {current}

    for step in range(len(grid) * len(grid[0])):
        if current == end:
            return path, explored, trace

        options = pick_best_neighbor(grid, current, end, mode, visited)
        if not options:
            break

        best = options[0]
        current = best[1]
        visited.add(current)
        path.append(current)
        explored.append(current)
        trace.append(
            {
                "step": step,
                "node": [current[0], current[1]],
                "difference_reduced": round(best[0], 3),
            }
        )

    return [], explored, trace


def solve_ao_star(
    grid: List[List[float]], start: Tuple[int, int], end: Tuple[int, int], mode: Heuristic
):
    path, explored, trace = solve_astar(grid, start, end, mode)
    for item in trace:
        item["and_or_note"] = "OR-choice on best frontier branch"
    return path, explored, trace


def solve_grid(payload: SolveRequest):
    grid = payload.grid
    start = tuple(payload.start)
    end = tuple(payload.end)
    validate_grid_request(grid, start, end)

    algo = payload.algorithm
    if algo == "bfs":
        path, explored, trace = solve_bfs(grid, start, end)
    elif algo == "dfs":
        path, explored, trace = solve_dfs(grid, start, end)
    elif algo == "dls":
        path, explored, trace = solve_dls(grid, start, end, payload.depth_limit)
    elif algo == "iddfs":
        path, explored, trace = solve_iddfs(grid, start, end, payload.depth_limit)
    elif algo == "ucs":
        path, explored, trace = solve_ucs(grid, start, end)
    elif algo == "bidirectional":
        path, explored, trace = solve_bidirectional(grid, start, end)
    elif algo == "greedy":
        path, explored, trace = solve_greedy(grid, start, end, payload.heuristic)
    elif algo == "astar":
        path, explored, trace = solve_astar(grid, start, end, payload.heuristic)
    elif algo == "idastar":
        path, explored, trace = solve_idastar(grid, start, end, payload.heuristic)
    elif algo == "beam":
        path, explored, trace = solve_beam(
            grid, start, end, payload.heuristic, payload.beam_width
        )
    elif algo == "hill_simple":
        path, explored, trace = solve_hill(grid, start, end, payload.heuristic, "simple")
    elif algo == "hill_steepest":
        path, explored, trace = solve_hill(grid, start, end, payload.heuristic, "steepest")
    elif algo == "hill_stochastic":
        path, explored, trace = solve_hill(
            grid, start, end, payload.heuristic, "stochastic"
        )
    elif algo == "simulated_annealing":
        path, explored, trace = solve_hill(grid, start, end, payload.heuristic, "annealing")
    elif algo == "local_beam":
        path, explored, trace = solve_local_beam(
            grid, start, end, payload.heuristic, payload.beam_width
        )
    elif algo == "genetic":
        path, explored, trace = solve_genetic(grid, start, end, payload.heuristic)
    elif algo == "means_end":
        path, explored, trace = solve_means_end(grid, start, end, payload.heuristic)
    else:  # ao_star
        path, explored, trace = solve_ao_star(grid, start, end, payload.heuristic)

    solved = bool(path)
    summary = (
        f"{algo.upper()} {'found a route' if solved else 'did not find a route'} from {start} to {end}."
    )
    return path, explored, trace, summary


def run_game_tree(prune: bool):
    leaves = [3, 5, 6, 9, 1, 2, 0, -1]
    visit_count = 0

    def minimax(depth: int, index: int, maximizing: bool, alpha: float, beta: float):
        nonlocal visit_count
        if depth == 3:
            visit_count += 1
            return leaves[index], [{"depth": depth, "leaf": index, "value": leaves[index]}]

        trace_local = []
        if maximizing:
            value = -float("inf")
            for i in [0, 1]:
                child_val, child_trace = minimax(
                    depth + 1, index * 2 + i, False, alpha, beta
                )
                trace_local.extend(child_trace)
                value = max(value, child_val)
                alpha = max(alpha, value)
                trace_local.append(
                    {
                        "depth": depth,
                        "node": index,
                        "type": "MAX",
                        "alpha": alpha,
                        "beta": beta,
                        "value": value,
                    }
                )
                if prune and beta <= alpha:
                    trace_local.append(
                        {
                            "depth": depth,
                            "node": index,
                            "pruned": True,
                        }
                    )
                    break
            return value, trace_local

        value = float("inf")
        for i in [0, 1]:
            child_val, child_trace = minimax(depth + 1, index * 2 + i, True, alpha, beta)
            trace_local.extend(child_trace)
            value = min(value, child_val)
            beta = min(beta, value)
            trace_local.append(
                {
                    "depth": depth,
                    "node": index,
                    "type": "MIN",
                    "alpha": alpha,
                    "beta": beta,
                    "value": value,
                }
            )
            if prune and beta <= alpha:
                trace_local.append(
                    {
                        "depth": depth,
                        "node": index,
                        "pruned": True,
                    }
                )
                break
        return value, trace_local

    value, trace = minimax(0, 0, True, -float("inf"), float("inf"))
    summary = (
        f"{'Alpha-Beta' if prune else 'Minimax'} selected utility {int(value)} after visiting {visit_count} leaves."
    )
    return summary, trace


def run_logic(which: str):
    facts = {"rain", "has_exam"}
    rules = [
        (("rain",), "cloudy"),
        (("cloudy", "has_exam"), "study_inside"),
        (("study_inside",), "pass_test"),
    ]
    goal = "pass_test"

    if which == "forward_chaining":
        known = set(facts)
        trace = []
        changed = True
        while changed:
            changed = False
            for premises, conclusion in rules:
                if all(p in known for p in premises) and conclusion not in known:
                    known.add(conclusion)
                    changed = True
                    trace.append(
                        {
                            "rule": f"{premises} -> {conclusion}",
                            "new_fact": conclusion,
                            "known_count": len(known),
                        }
                    )
        return (
            f"Forward chaining {'proved' if goal in known else 'did not prove'} {goal}.",
            trace,
        )

    if which == "backward_chaining":
        trace = []

        def prove(target: str, seen: set):
            if target in facts:
                trace.append({"goal": target, "status": "fact"})
                return True
            if target in seen:
                return False
            seen.add(target)
            for premises, conclusion in rules:
                if conclusion != target:
                    continue
                trace.append({"goal": target, "trying_rule": f"{premises}->{conclusion}"})
                if all(prove(p, seen) for p in premises):
                    return True
            return False

        ok = prove(goal, set())
        return f"Backward chaining {'proved' if ok else 'failed to prove'} {goal}.", trace

    clauses = [{"p", "q"}, {"~p", "r"}, {"~q"}, {"~r"}]
    trace = []

    def complementary(a: str, b: str) -> bool:
        return a == f"~{b}" or b == f"~{a}"

    changed = True
    while changed:
        changed = False
        snapshot = list(clauses)
        for i in range(len(snapshot)):
            for j in range(i + 1, len(snapshot)):
                c1, c2 = snapshot[i], snapshot[j]
                for lit1 in c1:
                    for lit2 in c2:
                        if complementary(lit1, lit2):
                            resolvent = (c1 | c2) - {lit1, lit2}
                            trace.append(
                                {
                                    "resolve": [sorted(c1), sorted(c2)],
                                    "on": [lit1, lit2],
                                    "resolvent": sorted(resolvent),
                                }
                            )
                            if not resolvent:
                                return "Resolution derived empty clause: contradiction proven.", trace
                            if resolvent not in clauses:
                                clauses.append(resolvent)
                                changed = True

    return "Resolution could not derive empty clause.", trace


def run_csp(which: str):
    variables = ["A", "B", "C", "D"]
    domains = {v: {"Red", "Green", "Blue"} for v in variables}
    edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")]

    def consistent(assign: Dict[str, str], v: str, color: str) -> bool:
        for a, b in edges:
            if a == v and b in assign and assign[b] == color:
                return False
            if b == v and a in assign and assign[a] == color:
                return False
        return True

    trace: List[dict] = []

    def backtrack(assign: Dict[str, str], use_fc: bool):
        if len(assign) == len(variables):
            return assign
        unassigned = [v for v in variables if v not in assign]
        var = unassigned[0]

        for color in sorted(domains[var]):
            trace.append({"var": var, "try": color, "assigned": len(assign)})
            if not consistent(assign, var, color):
                continue
            new_assign = dict(assign)
            new_assign[var] = color
            if use_fc:
                pruned = []
                failed = False
                for a, b in edges:
                    neigh = None
                    if a == var and b not in new_assign:
                        neigh = b
                    elif b == var and a not in new_assign:
                        neigh = a
                    if neigh and color in domains[neigh]:
                        domains[neigh].remove(color)
                        pruned.append((neigh, color))
                        if not domains[neigh]:
                            failed = True
                            break
                if not failed:
                    result = backtrack(new_assign, use_fc)
                    if result:
                        return result
                for neigh, c in pruned:
                    domains[neigh].add(c)
            else:
                result = backtrack(new_assign, use_fc)
                if result:
                    return result
        return None

    if which == "ac3":
        queue = deque(edges + [(b, a) for a, b in edges])

        def revise(x: str, y: str) -> bool:
            removed = False
            for vx in list(domains[x]):
                if not any(vx != vy for vy in domains[y]):
                    domains[x].remove(vx)
                    removed = True
            return removed

        while queue:
            x, y = queue.popleft()
            if revise(x, y):
                trace.append({"arc": [x, y], "domain": sorted(domains[x])})
                if not domains[x]:
                    return "AC-3 found inconsistency (empty domain).", trace
                for a, b in edges:
                    if b == x and a != y:
                        queue.append((a, x))
                    elif a == x and b != y:
                        queue.append((b, x))

        return "AC-3 enforced arc consistency on map-coloring CSP.", trace

    assignment = backtrack({}, use_fc=(which == "forward_checking"))
    label = "Forward checking" if which == "forward_checking" else "Backtracking"
    return f"{label} produced assignment: {assignment}", trace


def entropy(labels: List[str]) -> float:
    total = len(labels)
    counts = Counter(labels)
    result = 0.0
    for c in counts.values():
        p = c / total
        result -= p * math.log2(max(p, 1e-12))
    return result


def run_ml(which: str):
    dataset = [
        {"outlook": "sunny", "wind": "weak", "play": "yes"},
        {"outlook": "sunny", "wind": "strong", "play": "no"},
        {"outlook": "rain", "wind": "weak", "play": "yes"},
        {"outlook": "rain", "wind": "strong", "play": "no"},
        {"outlook": "overcast", "wind": "weak", "play": "yes"},
    ]

    if which == "id3":
        attrs = ["outlook", "wind"]
        labels = [d["play"] for d in dataset]
        base = entropy(labels)
        trace = []
        gains = {}
        for attr in attrs:
            subsets = {}
            for row in dataset:
                subsets.setdefault(row[attr], []).append(row["play"])
            weighted = 0.0
            for subset in subsets.values():
                weighted += (len(subset) / len(dataset)) * entropy(subset)
            gain = base - weighted
            gains[attr] = gain
            trace.append({"attribute": attr, "gain": round(gain, 4)})
        best = max(gains, key=gains.get)
        return f"ID3 picked root attribute '{best}'.", trace

    if which == "naive_bayes":
        counts = Counter([d["play"] for d in dataset])
        priors = {k: v / len(dataset) for k, v in counts.items()}
        query = {"outlook": "sunny", "wind": "weak"}
        trace = []
        scores = {}

        for cls in counts:
            prob = priors[cls]
            cls_rows = [d for d in dataset if d["play"] == cls]
            for feature, value in query.items():
                hit = sum(1 for row in cls_rows if row[feature] == value)
                likelihood = (hit + 1) / (len(cls_rows) + 3)
                prob *= likelihood
                trace.append(
                    {
                        "class": cls,
                        "feature": feature,
                        "value": value,
                        "likelihood": round(likelihood, 4),
                    }
                )
            scores[cls] = prob

        predicted = max(scores, key=scores.get)
        return f"Naive Bayes predicted class '{predicted}' for {query}.", trace

    points = [
        ((1.0, 1.0), "A"),
        ((1.2, 0.9), "A"),
        ((3.0, 3.0), "B"),
        ((3.2, 2.8), "B"),
        ((2.8, 3.3), "B"),
    ]
    query = (1.4, 1.2)
    k = 3
    dists = []
    for (x, y), label in points:
        d = math.dist((x, y), query)
        dists.append((d, label, (x, y)))
    dists.sort(key=lambda x: x[0])
    top = dists[:k]
    vote = Counter([lbl for _, lbl, _ in top]).most_common(1)[0][0]
    trace = [
        {"point": [p[0], p[1]], "label": lbl, "distance": round(d, 4)}
        for d, lbl, p in top
    ]
    return f"KNN predicted class '{vote}' for point {query} with k={k}.", trace


def run_prob(which: str):
    if which == "bayesian_inference":
        p_d = 0.01
        p_pos_d = 0.9
        p_pos_not = 0.05
        posterior = (p_pos_d * p_d) / ((p_pos_d * p_d) + (p_pos_not * (1 - p_d)))
        trace = [
            {"P(D)": p_d, "P(Pos|D)": p_pos_d, "P(Pos|~D)": p_pos_not},
            {"P(D|Pos)": round(posterior, 5)},
        ]
        return "Bayesian inference computed posterior probability after evidence.", trace

    if which == "bayesian_network":
        p_b = 0.001
        p_e = 0.002
        p_a_be = 0.95
        p_a_bne = 0.94
        p_a_nbe = 0.29
        p_a_nbne = 0.001
        p_a = (
            p_a_be * p_b * p_e
            + p_a_bne * p_b * (1 - p_e)
            + p_a_nbe * (1 - p_b) * p_e
            + p_a_nbne * (1 - p_b) * (1 - p_e)
        )
        trace = [
            {"B": p_b, "E": p_e},
            {"P(A)": round(p_a, 6), "network": "Burglary-Earthquake-Alarm"},
        ]
        return "Bayesian network marginal computed for Alarm node.", trace

    observations = ["walk", "shop", "clean"]
    states = ["Rainy", "Sunny"]
    start_p = {"Rainy": 0.6, "Sunny": 0.4}
    trans = {
        "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
        "Sunny": {"Rainy": 0.4, "Sunny": 0.6},
    }
    emit = {
        "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
        "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
    }

    V = [{}]
    path = {}
    for s in states:
        V[0][s] = start_p[s] * emit[s][observations[0]]
        path[s] = [s]

    trace = [{"t": 0, "probs": {k: round(v, 6) for k, v in V[0].items()}}]

    for t in range(1, len(observations)):
        V.append({})
        new_path = {}
        for s in states:
            prob, state = max(
                (
                    V[t - 1][s0] * trans[s0][s] * emit[s][observations[t]],
                    s0,
                )
                for s0 in states
            )
            V[t][s] = prob
            new_path[s] = path[state] + [s]
        path = new_path
        trace.append({"t": t, "probs": {k: round(v, 6) for k, v in V[t].items()}})

    best_state = max(states, key=lambda s: V[-1][s])
    return f"HMM (Viterbi) best hidden sequence: {path[best_state]}", trace


def run_strips():
    state = {"at_home", "have_money"}
    goal = {"have_food"}
    actions = [
        {
            "name": "go_store",
            "pre": {"at_home"},
            "add": {"at_store"},
            "del": {"at_home"},
        },
        {
            "name": "buy_food",
            "pre": {"at_store", "have_money"},
            "add": {"have_food"},
            "del": set(),
        },
    ]

    plan = []
    trace = []
    for _ in range(5):
        if goal.issubset(state):
            break
        applied = False
        for action in actions:
            if action["pre"].issubset(state) and not action["add"].issubset(state):
                state = (state - action["del"]) | action["add"]
                plan.append(action["name"])
                trace.append({"action": action["name"], "state": sorted(state)})
                applied = True
                break
        if not applied:
            break

    return f"STRIPS planning sequence: {plan}", trace


def solve_scenario(algo: ScenarioAlgorithm):
    if algo == "minimax":
        return run_game_tree(prune=False)
    if algo == "alpha_beta":
        return run_game_tree(prune=True)
    if algo in {"forward_chaining", "backward_chaining", "resolution"}:
        return run_logic(algo)
    if algo in {"csp_backtracking", "forward_checking", "ac3"}:
        mapped = "csp_backtracking" if algo == "csp_backtracking" else algo
        return run_csp(mapped)
    if algo in {"id3", "naive_bayes", "knn"}:
        return run_ml(algo)
    if algo in {"bayesian_inference", "bayesian_network", "hmm"}:
        return run_prob(algo)
    return run_strips()


@app.post("/solve", response_model=SolveResponse)
def solve(payload: SolveRequest):
    start_time = time.perf_counter()

    if payload.algorithm in GRID_ALGORITHMS:
        path, explored, trace, summary = solve_grid(payload)
    else:
        summary, trace = solve_scenario(payload.algorithm)  # type: ignore[arg-type]
        path = []
        explored = []

    solve_time_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "path": path,
        "explored": explored,
        "trace": trace,
        "summary": summary,
        "stats": {
            "nodes_explored": len(explored) if explored else len(trace),
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
        "terrain": sorted(GRID_ALGORITHMS),
        "scenario": [
            "minimax",
            "alpha_beta",
            "forward_chaining",
            "backward_chaining",
            "resolution",
            "csp_backtracking",
            "forward_checking",
            "ac3",
            "id3",
            "naive_bayes",
            "knn",
            "bayesian_inference",
            "bayesian_network",
            "hmm",
            "strips",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
