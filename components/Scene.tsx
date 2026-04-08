"use client";

import { Canvas, type ThreeEvent, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { motion } from "framer-motion";
import { createNoise2D } from "simplex-noise";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { usePathStore } from "../store/usePathStore";

type Algorithm =
  | "bfs"
  | "dfs"
  | "dls"
  | "iddfs"
  | "ucs"
  | "greedy"
  | "astar"
  | "hill_simple"
  | "hill_steepest"
  | "hill_stochastic";

type Heuristic = "manhattan" | "euclidean" | "diagonal";

type Stats = {
  nodes_explored: number;
  path_length: number;
  solve_time_ms: number;
};

type TraceStep = {
  step: number;
  node?: [number, number];
  [key: string]: string | number | boolean | [number, number] | undefined;
};

type TerrainNode = {
  x: number;
  z: number;
  elevation: number;
  passable: boolean;
};

type NodeCoord = [number, number];

type PickMode = "start" | "end" | null;

const GRID_SIZE = 50;
const WIDTH = 30;
const HALF = WIDTH / 2;
const NOISE_SCALE = 0.15;
const HEIGHT_SCALE = 3.5;

const HEURISTIC_ALGORITHMS = new Set<Algorithm>([
  "greedy",
  "astar",
  "hill_simple",
  "hill_steepest",
  "hill_stochastic",
]);

const DEPTH_ALGORITHMS = new Set<Algorithm>(["dls", "iddfs"]);

const ALGORITHM_GROUPS: Array<{ title: string; items: Array<{ value: Algorithm; label: string }> }> = [
  {
    title: "Uninformed Search",
    items: [
      { value: "bfs", label: "Breadth-First Search (BFS)" },
      { value: "dfs", label: "Depth-First Search (DFS)" },
      { value: "dls", label: "Depth-Limited Search (DLS)" },
      { value: "iddfs", label: "Iterative Deepening DFS (IDDFS)" },
      { value: "ucs", label: "Uniform Cost Search (UCS)" },
    ],
  },
  {
    title: "Informed Search",
    items: [
      { value: "greedy", label: "Greedy Best-First" },
      { value: "astar", label: "A*" },
    ],
  },
  {
    title: "Hill Climbing",
    items: [
      { value: "hill_simple", label: "Simple Hill Climbing" },
      { value: "hill_steepest", label: "Steepest Ascent Hill Climbing" },
      { value: "hill_stochastic", label: "Stochastic Hill Climbing" },
    ],
  },
];

export let terrainGridFlat: TerrainNode[] = [];

function ControlPanel({
  algorithm,
  heuristic,
  speed,
  depthLimit,
  startNode,
  endNode,
  pickMode,
  onAlgorithmChange,
  onHeuristicChange,
  onSpeedChange,
  onDepthLimitChange,
  onStartNodeChange,
  onEndNodeChange,
  onPickMode,
  onRegenerate,
  onRun,
  stats,
  summary,
  isLoading,
  error,
  traceLines,
}: {
  algorithm: Algorithm;
  heuristic: Heuristic;
  speed: number;
  depthLimit: number;
  startNode: NodeCoord;
  endNode: NodeCoord;
  pickMode: PickMode;
  onAlgorithmChange: (value: Algorithm) => void;
  onHeuristicChange: (value: Heuristic) => void;
  onSpeedChange: (value: number) => void;
  onDepthLimitChange: (value: number) => void;
  onStartNodeChange: (next: NodeCoord) => void;
  onEndNodeChange: (next: NodeCoord) => void;
  onPickMode: (mode: PickMode) => void;
  onRegenerate: () => void;
  onRun: () => void;
  stats: Stats | null;
  summary: string;
  isLoading: boolean;
  error: string | null;
  traceLines: string[];
}) {
  const showHeuristic = HEURISTIC_ALGORITHMS.has(algorithm);
  const showDepth = DEPTH_ALGORITHMS.has(algorithm);

  return (
    <motion.aside
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="panel"
    >
      <div className="panel-title">IAI Algorithm Lab</div>

      <div className="label">Algorithm</div>
      <select
        value={algorithm}
        onChange={(event) => onAlgorithmChange(event.target.value as Algorithm)}
        className="select"
      >
        {ALGORITHM_GROUPS.map((group) => (
          <optgroup key={group.title} label={group.title}>
            {group.items.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>

      <div className="mode-tag">Path-based terrain mode</div>

      {showHeuristic ? (
        <>
          <div className="label">Heuristic</div>
          <div className="pill-row">
            {(["manhattan", "euclidean", "diagonal"] as Heuristic[]).map((item) => (
              <button
                key={item}
                onClick={() => onHeuristicChange(item)}
                className={heuristic === item ? "pill pill-active" : "pill pill-inactive"}
              >
                {item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
          </div>
        </>
      ) : null}

      {showDepth ? (
        <>
          <div className="label">Depth Limit</div>
          <div className="speed-row">
            <input
              type="range"
              min={2}
              max={80}
              value={depthLimit}
              onChange={(event) => onDepthLimitChange(Number(event.target.value))}
            />
            <div className="speed-value">{depthLimit}</div>
          </div>
        </>
      ) : null}

      <div className="label">Animation Speed</div>
      <div className="speed-row">
        <input
          type="range"
          min={1}
          max={10}
          value={speed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
        />
        <div className="speed-value">{speed}</div>
      </div>

      <div className="label">Start Node (row, col)</div>
      <div className="coord-row">
        <input
          type="number"
          min={0}
          max={GRID_SIZE - 1}
          value={startNode[0]}
          onChange={(event) => onStartNodeChange([Number(event.target.value), startNode[1]])}
        />
        <input
          type="number"
          min={0}
          max={GRID_SIZE - 1}
          value={startNode[1]}
          onChange={(event) => onStartNodeChange([startNode[0], Number(event.target.value)])}
        />
        <button
          onClick={() => onPickMode(pickMode === "start" ? null : "start")}
          className={pickMode === "start" ? "mini-button mini-active" : "mini-button"}
        >
          Pick
        </button>
      </div>

      <div className="label">Target Node (row, col)</div>
      <div className="coord-row">
        <input
          type="number"
          min={0}
          max={GRID_SIZE - 1}
          value={endNode[0]}
          onChange={(event) => onEndNodeChange([Number(event.target.value), endNode[1]])}
        />
        <input
          type="number"
          min={0}
          max={GRID_SIZE - 1}
          value={endNode[1]}
          onChange={(event) => onEndNodeChange([endNode[0], Number(event.target.value)])}
        />
        <button
          onClick={() => onPickMode(pickMode === "end" ? null : "end")}
          className={pickMode === "end" ? "mini-button mini-active" : "mini-button"}
        >
          Pick
        </button>
      </div>
      {pickMode ? <div className="hint">Click a terrain cell to set {pickMode} node.</div> : null}

      <button onClick={onRegenerate} className="ghost-button">
        Regenerate Terrain
      </button>

      <button onClick={onRun} disabled={isLoading} className="cta-button">
        {isLoading ? "Running..." : "Run Algorithm"}
      </button>

      {stats ? (
        <div className="stats">
          <div>Nodes Explored: {stats.nodes_explored}</div>
          <div>Path Length: {stats.path_length}</div>
          <div>Solve Time: {stats.solve_time_ms}ms</div>
        </div>
      ) : null}

      {summary ? <div className="summary">{summary}</div> : null}
      {error ? <div className="error">{error}</div> : null}

      {traceLines.length ? (
        <div className="trace">
          {traceLines.map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      ) : null}

      <style jsx>{`
        .panel {
          background: rgba(255, 255, 250, 0.95);
          border: 1px solid rgba(30, 41, 59, 0.12);
          border-radius: 0;
          padding: 18px;
          width: 100%;
          height: 100%;
          max-height: none;
          overflow: auto;
          color: #111827;
          pointer-events: auto;
          box-shadow: none;
        }
        .panel-title {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.2em;
          color: #334155;
        }
        .label {
          margin-top: 12px;
          font-size: 13px;
          color: #1f2937;
        }
        .select {
          margin-top: 8px;
          width: 100%;
          border-radius: 8px;
          border: 1px solid rgba(51, 65, 85, 0.35);
          padding: 8px;
          font-size: 13px;
          color: #0f172a;
          background: #ffffff;
        }
        .mode-tag {
          margin-top: 8px;
          border-radius: 8px;
          font-size: 12px;
          padding: 8px;
          background: rgba(148, 163, 184, 0.18);
          color: #334155;
        }
        .pill-row {
          margin-top: 8px;
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .pill {
          padding: 6px 12px;
          border-radius: 999px;
          font-size: 11px;
          letter-spacing: 0.08em;
          border: 1px solid transparent;
        }
        .pill-active {
          background: #0f172a;
          color: #ffffff;
        }
        .pill-inactive {
          background: rgba(255, 255, 255, 0.82);
          color: #374151;
          border-color: rgba(51, 65, 85, 0.25);
        }
        .speed-row {
          margin-top: 8px;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .speed-value {
          font-size: 14px;
          color: #1f2937;
          width: 30px;
          text-align: right;
        }
        input[type="range"] {
          width: 100%;
          accent-color: #0f172a;
        }
        .coord-row {
          margin-top: 8px;
          display: grid;
          grid-template-columns: 1fr 1fr auto;
          gap: 8px;
        }
        .coord-row input {
          border-radius: 8px;
          border: 1px solid rgba(51, 65, 85, 0.3);
          padding: 8px;
          font-size: 12px;
        }
        .mini-button {
          border-radius: 8px;
          border: 1px solid rgba(51, 65, 85, 0.4);
          background: #ffffff;
          color: #334155;
          font-size: 12px;
          padding: 6px 10px;
        }
        .mini-active {
          background: #0f172a;
          color: #ffffff;
        }
        .hint {
          margin-top: 6px;
          font-size: 11px;
          color: #0f766e;
        }
        .ghost-button {
          margin-top: 14px;
          width: 100%;
          border-radius: 8px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          padding: 10px 12px;
          font-size: 13px;
          color: #1f2937;
          background: rgba(255, 255, 255, 0.75);
        }
        .cta-button {
          margin-top: 10px;
          width: 100%;
          border-radius: 8px;
          border: none;
          background: #0f172a;
          color: #ffffff;
          padding: 10px 12px;
          font-size: 14px;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }
        .stats {
          margin-top: 12px;
          font-family: "Courier New", monospace;
          font-size: 12px;
          color: #1f2937;
          display: grid;
          gap: 3px;
        }
        .summary {
          margin-top: 10px;
          font-size: 12px;
          color: #0f172a;
          border-radius: 8px;
          padding: 8px;
          background: rgba(15, 23, 42, 0.06);
        }
        .error {
          margin-top: 10px;
          font-size: 12px;
          color: #b3261e;
        }
        .trace {
          margin-top: 10px;
          max-height: 170px;
          overflow: auto;
          padding: 10px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.92);
          border: 1px solid rgba(51, 65, 85, 0.28);
          font-family: "Courier New", monospace;
          font-size: 11px;
          color: #111827;
          display: grid;
          gap: 3px;
        }
      `}</style>
    </motion.aside>
  );
}

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateTerrain(seed: number) {
  const rand = mulberry32(seed);
  const noise2D = createNoise2D(rand);
  const positions: number[] = [];
  const colors: number[] = [];
  const elevations: number[][] = [];
  const flat: TerrainNode[] = [];

  let minY = Infinity;
  let maxY = -Infinity;

  for (let row = 0; row < GRID_SIZE; row += 1) {
    elevations[row] = [];
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const x = (col / (GRID_SIZE - 1) - 0.5) * WIDTH;
      const z = (row / (GRID_SIZE - 1) - 0.5) * WIDTH;
      const y = noise2D(x * NOISE_SCALE, z * NOISE_SCALE) * HEIGHT_SCALE;
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      elevations[row][col] = y;
      positions.push(x, y, z);
    }
  }

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const y = elevations[row][col];
      const t = (y - minY) / (maxY - minY || 1);
      const cold = new THREE.Color("#7a7d85");
      const hot = new THREE.Color("#d39a45");
      const color = cold.clone().lerp(hot, t);
      colors.push(color.r, color.g, color.b);

      const x = (col / (GRID_SIZE - 1) - 0.5) * WIDTH;
      const z = (row / (GRID_SIZE - 1) - 0.5) * WIDTH;
      const elevationNormalized = (y - minY) / (maxY - minY || 1);
      const passable = elevationNormalized >= 0.15;
      flat.push({ x, z, elevation: elevationNormalized, passable });
    }
  }

  const gridForSolver = elevations.map((row) =>
    row.map((y) => {
      const elevationNormalized = (y - minY) / (maxY - minY || 1);
      return elevationNormalized >= 0.15 ? elevationNormalized : 0;
    })
  );

  terrainGridFlat = flat;

  return {
    positions: new Float32Array(positions),
    colors: new Float32Array(colors),
    elevations,
    gridForSolver,
  };
}

function buildGeometry(positions: Float32Array, colors: Float32Array) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const indices: number[] = [];
  for (let row = 0; row < GRID_SIZE - 1; row += 1) {
    for (let col = 0; col < GRID_SIZE - 1; col += 1) {
      const a = row * GRID_SIZE + col;
      const b = row * GRID_SIZE + col + 1;
      const c = (row + 1) * GRID_SIZE + col;
      const d = (row + 1) * GRID_SIZE + col + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

const defaultEmissive = new THREE.Color("#4b5563");
const exploredMid = new THREE.Color("#ff1744");
const exploredHot = new THREE.Color("#00c853");
const blockedColor = new THREE.Color("#d92d20");

function rgbTraceColor(t: number) {
  const start = new THREE.Color("#ff1744");
  const middle = new THREE.Color("#00c853");
  const end = new THREE.Color("#2979ff");
  if (t < 0.5) {
    return start.clone().lerp(middle, t / 0.5);
  }
  return middle.clone().lerp(end, (t - 0.5) / 0.5);
}

function toPoint(row: number, col: number, terrain: ReturnType<typeof generateTerrain>, lift = 0.18) {
  const x = (col / (GRID_SIZE - 1) - 0.5) * WIDTH;
  const z = (row / (GRID_SIZE - 1) - 0.5) * WIDTH;
  const y = terrain.elevations[row][col] + lift;
  return new THREE.Vector3(x, y, z);
}

function TerrainScene({
  terrain,
  speed,
  solverTrace,
  startNode,
  endNode,
  pickMode,
  onPickNode,
  onTraceStep,
}: {
  terrain: ReturnType<typeof generateTerrain>;
  speed: number;
  solverTrace: TraceStep[];
  startNode: NodeCoord;
  endNode: NodeCoord;
  pickMode: PickMode;
  onPickNode: (row: number, col: number) => void;
  onTraceStep: (line: string) => void;
}) {
  const instancedRef = useRef<THREE.InstancedMesh>(null);
  const pathMeshRef = useRef<THREE.Mesh>(null);
  const intervalRef = useRef<number | null>(null);
  const animationRef = useRef<number | null>(null);

  const { grid, explored, path, setGrid } = usePathStore();

  const instanceStates = useRef({
    locked: Array(GRID_SIZE * GRID_SIZE).fill(null) as Array<THREE.Color | null>,
    animating: new Set<number>(),
    pulseOffsets: Array(GRID_SIZE * GRID_SIZE)
      .fill(0)
      .map((_, index) => (index % 17) * 0.2),
  });

  const tracePointsRef = useRef<THREE.Vector3[]>([]);
  const [tracePoints, setTracePoints] = useState<THREE.Vector3[]>([]);
  const [traceColors, setTraceColors] = useState<THREE.Color[]>([]);

  const geometry = useMemo(() => buildGeometry(terrain.positions, terrain.colors), [terrain]);
  const wireGeometry = useMemo(() => new THREE.WireframeGeometry(geometry), [geometry]);

  const startMarker = useMemo(() => toPoint(startNode[0], startNode[1], terrain, 0.45), [startNode, terrain]);
  const endMarker = useMemo(() => toPoint(endNode[0], endNode[1], terrain, 0.45), [endNode, terrain]);

  useEffect(() => {
    setGrid(terrain.gridForSolver);

    const instanced = instancedRef.current;
    if (!instanced) return;

    const tempObject = new THREE.Object3D();
    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let col = 0; col < GRID_SIZE; col += 1) {
        const index = row * GRID_SIZE + col;
        const x = (col / (GRID_SIZE - 1) - 0.5) * WIDTH;
        const z = (row / (GRID_SIZE - 1) - 0.5) * WIDTH;
        const y = terrain.elevations[row][col];
        tempObject.position.set(x, y, z);
        tempObject.updateMatrix();
        instanced.setMatrixAt(index, tempObject.matrix);
        instanced.setColorAt(index, defaultEmissive);
      }
    }

    instanced.instanceMatrix.needsUpdate = true;
    if (instanced.instanceColor) {
      instanced.instanceColor.needsUpdate = true;
    }

    instanceStates.current.locked.fill(null);
    instanceStates.current.animating.clear();
    tracePointsRef.current = [];
    setTracePoints([]);
    setTraceColors([]);

    if (pathMeshRef.current?.geometry) {
      pathMeshRef.current.geometry.dispose();
    }
  }, [terrain, setGrid]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
      if (animationRef.current) window.cancelAnimationFrame(animationRef.current);
    };
  }, []);

  useFrame(({ clock }) => {
    const instanced = instancedRef.current;
    if (!instanced || !instanced.instanceColor) return;

    const time = clock.getElapsedTime();
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i += 1) {
      if (instanceStates.current.locked[i] || instanceStates.current.animating.has(i)) continue;
      const pulse = 0.55 + 0.45 * Math.sin((time * Math.PI * 2) / 2 + instanceStates.current.pulseOffsets[i]);
      const color = defaultEmissive.clone().multiplyScalar(pulse);
      instanced.setColorAt(i, color);
    }
    instanced.instanceColor.needsUpdate = true;
  });

  const getIndex = (row: number, col: number) => row * GRID_SIZE + col;

  const setInstanceColor = (index: number, color: THREE.Color) => {
    const instanced = instancedRef.current;
    if (!instanced) return;
    instanced.setColorAt(index, color);
    if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
  };

  const animateColor = (index: number) => {
    const instanced = instancedRef.current;
    if (!instanced) return;
    instanceStates.current.animating.add(index);

    const start = performance.now();
    const duration = 300;
    const cold = defaultEmissive.clone();
    const mid = exploredMid.clone();
    const hot = exploredHot.clone();

    const tick = (now: number) => {
      const elapsed = now - start;
      const t = Math.min(elapsed / duration, 1);
      const color = t < 0.5 ? cold.clone().lerp(mid, t / 0.5) : mid.clone().lerp(hot, (t - 0.5) / 0.5);
      instanced.setColorAt(index, color);
      if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
      if (t < 1) {
        animationRef.current = requestAnimationFrame(tick);
      } else {
        instanceStates.current.animating.delete(index);
        instanceStates.current.locked[index] = hot;
      }
    };

    animationRef.current = requestAnimationFrame(tick);
  };

  const buildPathTube = (points: THREE.Vector3[]) => {
    if (!pathMeshRef.current) return;
    if (pathMeshRef.current.geometry) pathMeshRef.current.geometry.dispose();
    const curve = new THREE.CatmullRomCurve3(points);
    const geo = new THREE.TubeGeometry(curve, 120, 0.06, 8, false);

    const colorArray = new Float32Array(geo.attributes.position.count * 3);
    for (let i = 0; i < geo.attributes.position.count; i += 1) {
      const t = i / Math.max(geo.attributes.position.count - 1, 1);
      const color = rgbTraceColor(t);
      colorArray[i * 3 + 0] = color.r;
      colorArray[i * 3 + 1] = color.g;
      colorArray[i * 3 + 2] = color.b;
    }

    geo.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
    geo.setDrawRange(0, 0);
    pathMeshRef.current.geometry = geo;

    const totalCount = geo.index ? geo.index.count : geo.attributes.position.count;
    const start = performance.now();
    const duration = 800;
    const animate = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      geo.setDrawRange(0, Math.floor(totalCount * t));
      if (t < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  };

  const updateTraceLine = (row: number, col: number, stepIndex: number, total: number) => {
    tracePointsRef.current.push(toPoint(row, col, terrain, 0.25));

    const pointSnapshot = [...tracePointsRef.current];
    setTracePoints(pointSnapshot);
    setTraceColors(
      pointSnapshot.map((_, i) => rgbTraceColor(i / Math.max(pointSnapshot.length - 1, 1)))
    );

    const event = solverTrace[stepIndex];
    if (event) {
      const details = Object.entries(event)
        .filter(([key]) => key !== "node" && key !== "step")
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(" ");
      const nodeLabel = event.node ? `node(${event.node[0]},${event.node[1]})` : "scenario-step";
      onTraceStep(`#${event.step} ${nodeLabel} ${details}`);
    } else {
      onTraceStep(`#${stepIndex} node(${row},${col}) explored=${stepIndex + 1}/${total}`);
    }
  };

  useEffect(() => {
    if (!explored.length || !instancedRef.current) return;

    if (intervalRef.current) window.clearInterval(intervalRef.current);

    let index = 0;
    intervalRef.current = window.setInterval(() => {
      if (index >= explored.length) {
        if (intervalRef.current) window.clearInterval(intervalRef.current);
        if (path.length) {
          const points = path.map(([row, col]) => toPoint(row, col, terrain, 0.15));
          buildPathTube(points);
        }
        return;
      }

      const [row, col] = explored[index];
      animateColor(getIndex(row, col));
      updateTraceLine(row, col, index, explored.length);
      index += 1;
    }, (11 - speed) * 20);

    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
    };
  }, [explored, path, speed, terrain, solverTrace, onTraceStep]);

  const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
    const point = event.point as THREE.Vector3;
    if (!point) return;

    const col = Math.round(((point.x + HALF) / WIDTH) * (GRID_SIZE - 1));
    const row = Math.round(((point.z + HALF) / WIDTH) * (GRID_SIZE - 1));
    if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) return;

    if (pickMode) {
      onPickNode(row, col);
      return;
    }

    if ((row === startNode[0] && col === startNode[1]) || (row === endNode[0] && col === endNode[1])) {
      return;
    }

    const updatedGrid = grid.map((r) => r.slice());
    const baseElevation = terrain.gridForSolver[row][col];
    const index = getIndex(row, col);
    const currentlyBlocked = updatedGrid[row][col] === 0;

    if (currentlyBlocked) {
      updatedGrid[row][col] = baseElevation || 0.2;
      setInstanceColor(index, defaultEmissive);
      instanceStates.current.locked[index] = null;
    } else {
      updatedGrid[row][col] = 0;
      setInstanceColor(index, blockedColor);
      instanceStates.current.locked[index] = blockedColor;
    }

    setGrid(updatedGrid);

    const flatIndex = row * GRID_SIZE + col;
    const flatNode = terrainGridFlat[flatIndex];
    if (flatNode) {
      flatNode.passable = updatedGrid[row][col] !== 0;
      flatNode.elevation = updatedGrid[row][col];
    }
  };

  return (
    <>
      <mesh geometry={geometry} onPointerDown={handlePointerDown} receiveShadow>
        <meshStandardMaterial vertexColors roughness={0.6} metalness={0.1} />
      </mesh>

      <lineSegments geometry={wireGeometry}>
        <lineBasicMaterial color="#4b5563" opacity={0.32} transparent />
      </lineSegments>

      {tracePoints.length > 1 ? (
        <Line points={tracePoints} vertexColors={traceColors} lineWidth={2} transparent opacity={0.95} />
      ) : null}

      <instancedMesh ref={instancedRef} args={[undefined, undefined, GRID_SIZE * GRID_SIZE]}>
        <sphereGeometry args={[0.08, 10, 10]} />
        <meshStandardMaterial color="#ffffff" emissive="#4b5563" emissiveIntensity={1.2} vertexColors />
      </instancedMesh>

      <mesh ref={pathMeshRef}>
        <meshStandardMaterial
          color="#ffffff"
          emissive="#ffffff"
          emissiveIntensity={0.45}
          vertexColors
          transparent
          opacity={0.9}
        />
      </mesh>

      <mesh position={startMarker}>
        <sphereGeometry args={[0.24, 18, 18]} />
        <meshStandardMaterial color="#1d4ed8" emissive="#1d4ed8" emissiveIntensity={0.65} />
      </mesh>

      <mesh position={endMarker}>
        <sphereGeometry args={[0.24, 18, 18]} />
        <meshStandardMaterial color="#b91c1c" emissive="#b91c1c" emissiveIntensity={0.65} />
      </mesh>

      <OrbitControls enableDamping dampingFactor={0.08} minPolarAngle={0.2} maxPolarAngle={1.2} />
    </>
  );
}

export default function Scene() {
  const { grid, stats, summary, isRunning, setResult, setRunning, reset, setGrid } = usePathStore();
  const [seed, setSeed] = useState(42);
  const [terrain, setTerrain] = useState(() => generateTerrain(seed));
  const [algorithm, setAlgorithm] = useState<Algorithm>("astar");
  const [heuristic, setHeuristic] = useState<Heuristic>("euclidean");
  const [speed, setSpeed] = useState(5);
  const [depthLimit, setDepthLimit] = useState(20);
  const [startNode, setStartNode] = useState<NodeCoord>([0, 0]);
  const [endNode, setEndNode] = useState<NodeCoord>([GRID_SIZE - 1, GRID_SIZE - 1]);
  const [pickMode, setPickMode] = useState<PickMode>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [solverTrace, setSolverTrace] = useState<TraceStep[]>([]);
  const [traceLines, setTraceLines] = useState<string[]>([]);

  const fog = useMemo(() => new THREE.Fog("#f1efe8", 20, 80), []);

  const ensureNodePassable = useCallback(
    (node: NodeCoord) => {
      const [row, col] = node;
      if (!grid.length || !grid[row] || grid[row][col] !== 0) return;

      const updatedGrid = grid.map((r) => r.slice());
      const base = terrain.gridForSolver[row][col] || 0.2;
      updatedGrid[row][col] = base;
      setGrid(updatedGrid);

      const idx = row * GRID_SIZE + col;
      const flatNode = terrainGridFlat[idx];
      if (flatNode) {
        flatNode.passable = true;
        flatNode.elevation = base;
      }
    },
    [grid, setGrid, terrain]
  );

  useEffect(() => {
    ensureNodePassable(startNode);
    ensureNodePassable(endNode);
  }, [startNode, endNode, ensureNodePassable]);

  const clampNode = useCallback((node: NodeCoord): NodeCoord => {
    const row = Math.max(0, Math.min(GRID_SIZE - 1, node[0]));
    const col = Math.max(0, Math.min(GRID_SIZE - 1, node[1]));
    return [row, col];
  }, []);

  const handlePickNode = useCallback(
    (row: number, col: number) => {
      const next: NodeCoord = [row, col];
      if (pickMode === "start") {
        setStartNode(next);
      } else if (pickMode === "end") {
        setEndNode(next);
      }
      setPickMode(null);
    },
    [pickMode]
  );

  const handleTraceStep = useCallback((line: string) => {
    setTraceLines((previous) => [line, ...previous].slice(0, 14));
  }, []);

  const handleRun = async () => {
    if (!grid.length || isRunning) return;

    const start = clampNode(startNode);
    const end = clampNode(endNode);

    reset();
    setRunError(null);
    setSolverTrace([]);
    setTraceLines([]);
    setRunning(true);

    try {
      const defaultApiBase =
        typeof window !== "undefined" && window.location.hostname.endsWith("vercel.app")
          ? `${window.location.origin}/_/backend`
          : "http://127.0.0.1:8000";

      const apiBase = (process.env.NEXT_PUBLIC_API_URL || defaultApiBase).replace(/\/$/, "");

      const response = await fetch(`${apiBase}/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grid,
          start,
          end,
          algorithm,
          heuristic,
          depth_limit: depthLimit,
        }),
      });

      if (!response.ok) {
        throw new Error(`Solver request failed (${response.status})`);
      }

      const data = await response.json();
      const trace: TraceStep[] = Array.isArray(data.trace) ? data.trace : [];
      setSolverTrace(trace);

      setResult({
        path: data.path || [],
        explored: data.explored || [],
        stats: data.stats,
        summary: data.summary || "",
      });

      if (!Array.isArray(data.explored) || data.explored.length === 0) {
        const quickLines = trace
          .slice(0, 12)
          .map((item) => JSON.stringify(item))
          .reverse();
        setTraceLines(quickLines);
      }
    } catch (error) {
      setRunning(false);
      setRunError("Search request failed. Check backend is running on :8000.");
      console.error(error);
    }
  };

  const handleRegenerate = () => {
    reset();
    setRunError(null);
    setSolverTrace([]);
    setTraceLines([]);
    setPickMode(null);

    const nextSeed = seed + 1;
    setSeed(nextSeed);
    const nextTerrain = generateTerrain(nextSeed);
    setTerrain(nextTerrain);

    setStartNode([0, 0]);
    setEndNode([GRID_SIZE - 1, GRID_SIZE - 1]);
  };

  return (
    <div className="layout-root">
      <aside className="layout-sidebar">
        <ControlPanel
          algorithm={algorithm}
          heuristic={heuristic}
          speed={speed}
          depthLimit={depthLimit}
          startNode={startNode}
          endNode={endNode}
          pickMode={pickMode}
          onAlgorithmChange={setAlgorithm}
          onHeuristicChange={setHeuristic}
          onSpeedChange={setSpeed}
          onDepthLimitChange={setDepthLimit}
          onStartNodeChange={(next) => setStartNode(clampNode(next))}
          onEndNodeChange={(next) => setEndNode(clampNode(next))}
          onPickMode={setPickMode}
          onRegenerate={handleRegenerate}
          onRun={handleRun}
          stats={stats}
          summary={summary}
          isLoading={isRunning}
          error={runError}
          traceLines={traceLines}
        />
      </aside>

      <main className="layout-sim">
        <Canvas
          dpr={[1, 2]}
          camera={{ position: [0, 18, 22], fov: 55 }}
          style={{
            background:
              "radial-gradient(circle at 30% 20%, #f8f7f2 0%, #ebe9df 50%, #d9d6c8 100%)",
          }}
        >
          <primitive attach="fog" object={fog} />
          <ambientLight intensity={0.3} />
          <directionalLight position={[0, 20, 0]} intensity={1} />

          <TerrainScene
            terrain={terrain}
            speed={speed}
            solverTrace={solverTrace}
            startNode={startNode}
            endNode={endNode}
            pickMode={pickMode}
            onPickNode={handlePickNode}
            onTraceStep={handleTraceStep}
          />
        </Canvas>
      </main>

      <style jsx>{`
        .layout-root {
          width: 100vw;
          height: 100vh;
          display: grid;
          grid-template-columns: 380px minmax(0, 1fr);
          background: #ece9df;
        }
        .layout-sidebar {
          min-width: 0;
          border-right: 1px solid rgba(30, 41, 59, 0.12);
          background: #f7f5ee;
          overflow: hidden;
        }
        .layout-sim {
          min-width: 0;
          height: 100%;
        }
        @media (max-width: 980px) {
          .layout-root {
            grid-template-columns: 1fr;
            grid-template-rows: auto minmax(0, 1fr);
          }
          .layout-sidebar {
            max-height: 48vh;
            border-right: none;
            border-bottom: 1px solid rgba(30, 41, 59, 0.12);
          }
        }
      `}</style>
    </div>
  );
}
