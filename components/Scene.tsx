"use client";

import { Canvas, type ThreeEvent, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { motion } from "framer-motion";
import { createNoise2D } from "simplex-noise";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { usePathStore } from "../store/usePathStore";

type Algorithm = "astar" | "greedy" | "bfs" | "dfs";

type Heuristic = "manhattan" | "euclidean" | "diagonal";

type Stats = {
  nodes_explored: number;
  path_length: number;
  solve_time_ms: number;
};

type TraceStep = {
  step: number;
  node: [number, number];
  g?: number;
  h?: number;
  f?: number;
  open_size?: number;
  closed_size?: number;
  frontier_size?: number;
  visited_size?: number;
};

type TerrainNode = {
  x: number;
  z: number;
  elevation: number;
  passable: boolean;
};

const GRID_SIZE = 50;
const WIDTH = 30;
const HALF = WIDTH / 2;
const NOISE_SCALE = 0.15;
const HEIGHT_SCALE = 3.5;

export let terrainGridFlat: TerrainNode[] = [];

function ControlPanel({
  algorithm,
  heuristic,
  speed,
  onAlgorithmChange,
  onHeuristicChange,
  onSpeedChange,
  onRegenerate,
  onRun,
  stats,
  isLoading,
  error,
  traceLines,
}: {
  algorithm: Algorithm;
  heuristic: Heuristic;
  speed: number;
  onAlgorithmChange: (value: Algorithm) => void;
  onHeuristicChange: (value: Heuristic) => void;
  onSpeedChange: (value: number) => void;
  onRegenerate: () => void;
  onRun: () => void;
  stats: Stats | null;
  isLoading: boolean;
  error: string | null;
  traceLines: string[];
}) {
  const showHeuristic = useMemo(
    () => algorithm === "astar" || algorithm === "greedy",
    [algorithm]
  );

  return (
    <motion.aside
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="panel"
    >
      <div className="panel-title">PathMind Control</div>

      <div className="label">Algorithm</div>
      <div className="pill-row">
        {(["astar", "greedy", "bfs", "dfs"] as Algorithm[]).map((item) => (
          <button
            key={item}
            onClick={() => onAlgorithmChange(item)}
            className={
              algorithm === item ? "pill pill-active" : "pill pill-inactive"
            }
          >
            {item.toUpperCase() === "ASTAR" ? "A*" : item.toUpperCase()}
          </button>
        ))}
      </div>

      {showHeuristic ? (
        <>
          <div className="label">Heuristic</div>
          <div className="pill-row">
            {(["manhattan", "euclidean", "diagonal"] as Heuristic[]).map(
              (item) => (
                <button
                  key={item}
                  onClick={() => onHeuristicChange(item)}
                  className={
                    heuristic === item
                      ? "pill pill-active"
                      : "pill pill-inactive"
                  }
                >
                  {item[0].toUpperCase() + item.slice(1)}
                </button>
              )
            )}
          </div>
        </>
      ) : null}

      <div className="label">Speed</div>
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

      <button onClick={onRegenerate} className="ghost-button">
        Regenerate Terrain
      </button>

      <button onClick={onRun} disabled={isLoading} className="cta-button">
        {isLoading ? "Running..." : "Run Search"}
      </button>

      {stats ? (
        <div className="stats">
          <div>Nodes Explored: {stats.nodes_explored}</div>
          <div>Path Length: {stats.path_length}</div>
          <div>Solve Time: {stats.solve_time_ms}ms</div>
        </div>
      ) : null}
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
          background: rgba(255, 255, 250, 0.88);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(30, 41, 59, 0.18);
          border-radius: 16px;
          padding: 24px;
          width: 280px;
          color: #111827;
          pointer-events: auto;
          box-shadow: 0 18px 44px rgba(30, 41, 59, 0.16);
        }
        .panel-title {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.2em;
          color: #334155;
        }
        .label {
          margin-top: 16px;
          font-size: 14px;
          color: #1f2937;
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
          transition: color 0.2s ease, background 0.2s ease, border 0.2s ease;
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
        input[type="range"] {
          width: 100%;
          accent-color: #0f172a;
        }
        .speed-value {
          font-size: 14px;
          color: #1f2937;
          width: 24px;
          text-align: right;
        }
        .ghost-button {
          margin-top: 16px;
          width: 100%;
          border-radius: 8px;
          border: 1px solid rgba(51, 65, 85, 0.5);
          padding: 10px 12px;
          font-size: 13px;
          color: #1f2937;
          background: rgba(255, 255, 255, 0.75);
          transition: border 0.2s ease;
        }
        .ghost-button:hover {
          border-color: #0f172a;
        }
        .cta-button {
          margin-top: 12px;
          width: 100%;
          border-radius: 8px;
          border: none;
          background: #0f172a;
          color: #ffffff;
          padding: 10px 12px;
          font-size: 16px;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }
        .cta-button:disabled {
          opacity: 0.6;
        }
        .stats {
          margin-top: 16px;
          font-family: "Courier New", monospace;
          font-size: 13px;
          color: #1f2937;
          display: grid;
          gap: 4px;
        }
        .error {
          margin-top: 10px;
          font-size: 12px;
          color: #b3261e;
        }
        .trace {
          margin-top: 12px;
          max-height: 170px;
          overflow: auto;
          padding: 10px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.92);
          border: 1px solid rgba(51, 65, 85, 0.28);
          font-family: "Courier New", monospace;
          font-size: 12px;
          color: #111827;
          display: grid;
          gap: 4px;
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

function TerrainScene({
  terrain,
  speed,
  solverTrace,
  onTraceStep,
}: {
  terrain: ReturnType<typeof generateTerrain>;
  speed: number;
  solverTrace: TraceStep[];
  onTraceStep: (line: string) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const wireRef = useRef<THREE.LineSegments>(null);
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

  const geometry = useMemo(
    () => buildGeometry(terrain.positions, terrain.colors),
    [terrain]
  );

  const wireGeometry = useMemo(() => new THREE.WireframeGeometry(geometry), [
    geometry,
  ]);

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

    if (pathMeshRef.current && pathMeshRef.current.geometry) {
      pathMeshRef.current.geometry.dispose();
    }
    tracePointsRef.current = [];
    setTracePoints([]);
    setTraceColors([]);
  }, [terrain, setGrid]);

  useEffect(() => {
    if (explored.length !== 0 || path.length !== 0) return;
    tracePointsRef.current = [];
    setTracePoints([]);
    setTraceColors([]);
    if (pathMeshRef.current && pathMeshRef.current.geometry) {
      pathMeshRef.current.geometry.dispose();
    }
  }, [explored.length, path.length]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
      if (animationRef.current) {
        window.cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  useFrame(({ clock }) => {
    const instanced = instancedRef.current;
    if (!instanced || !instanced.instanceColor) return;

    const time = clock.getElapsedTime();
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i += 1) {
      if (instanceStates.current.locked[i]) {
        continue;
      }
      if (instanceStates.current.animating.has(i)) {
        continue;
      }
      const pulse =
        0.55 +
        0.45 *
          Math.sin(
            (time * Math.PI * 2) / 2 + instanceStates.current.pulseOffsets[i]
          );
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
    if (instanced.instanceColor) {
      instanced.instanceColor.needsUpdate = true;
    }
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
      let color: THREE.Color;
      if (t < 0.5) {
        const tt = t / 0.5;
        color = cold.clone().lerp(mid, tt);
      } else {
        const tt = (t - 0.5) / 0.5;
        color = mid.clone().lerp(hot, tt);
      }
      instanced.setColorAt(index, color);
      if (instanced.instanceColor) {
        instanced.instanceColor.needsUpdate = true;
      }
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
    if (pathMeshRef.current.geometry) {
      pathMeshRef.current.geometry.dispose();
    }
    const curve = new THREE.CatmullRomCurve3(points);
    const geometry = new THREE.TubeGeometry(curve, 120, 0.06, 8, false);
    const colorArray = new Float32Array(
      geometry.attributes.position.count * 3
    );
    for (let i = 0; i < geometry.attributes.position.count; i += 1) {
      const t = i / Math.max(geometry.attributes.position.count - 1, 1);
      const color = rgbTraceColor(t);
      colorArray[i * 3 + 0] = color.r;
      colorArray[i * 3 + 1] = color.g;
      colorArray[i * 3 + 2] = color.b;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
    geometry.setDrawRange(0, 0);
    pathMeshRef.current.geometry = geometry;

    const totalCount = geometry.index
      ? geometry.index.count
      : geometry.attributes.position.count;

    const start = performance.now();
    const duration = 800;
    const animate = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      geometry.setDrawRange(0, Math.floor(totalCount * t));
      if (t < 1) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);
  };

  const updateTraceLine = (
    row: number,
    col: number,
    stepIndex: number,
    total: number
  ) => {
    const x = (col / (GRID_SIZE - 1) - 0.5) * WIDTH;
    const z = (row / (GRID_SIZE - 1) - 0.5) * WIDTH;
    const y = terrain.elevations[row][col] + 0.25;
    tracePointsRef.current.push(new THREE.Vector3(x, y, z));

    const pointSnapshot = [...tracePointsRef.current];
    setTracePoints(pointSnapshot);
    setTraceColors(
      pointSnapshot.map((_, i) =>
        rgbTraceColor(i / Math.max(pointSnapshot.length - 1, 1))
      )
    );

    const event = solverTrace[stepIndex];
    if (event) {
      const details = Object.entries(event)
        .filter(([key]) => key !== "node" && key !== "step")
        .map(([key, value]) => `${key}=${value}`)
        .join(" ");
      onTraceStep(
        `#${event.step} node(${event.node[0]},${event.node[1]}) ${details}`
      );
    } else {
      onTraceStep(
        `#${stepIndex} node(${row},${col}) explored=${stepIndex + 1}/${total}`
      );
    }
  };

  useEffect(() => {
    if (!explored.length) return;
    if (!instancedRef.current) return;

    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
    }

    let index = 0;
    intervalRef.current = window.setInterval(() => {
      if (index >= explored.length) {
        if (intervalRef.current) {
          window.clearInterval(intervalRef.current);
        }
        if (path.length) {
          const points = path.map(([row, col]) => {
            const x = (col / (GRID_SIZE - 1) - 0.5) * WIDTH;
            const z = (row / (GRID_SIZE - 1) - 0.5) * WIDTH;
            const y = terrain.elevations[row][col] + 0.15;
            return new THREE.Vector3(x, y, z);
          });
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
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
    };
  }, [explored, path, speed, terrain.elevations, onTraceStep, solverTrace]);

  const handleToggleObstacle = (event: ThreeEvent<PointerEvent>) => {
    if (!event.point) return;
    const point = event.point as THREE.Vector3;
    const col = Math.round(((point.x + HALF) / WIDTH) * (GRID_SIZE - 1));
    const row = Math.round(((point.z + HALF) / WIDTH) * (GRID_SIZE - 1));
    if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) return;

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
      <mesh
        ref={meshRef}
        geometry={geometry}
        onPointerDown={handleToggleObstacle}
        receiveShadow
      >
        <meshStandardMaterial
          vertexColors
          roughness={0.6}
          metalness={0.1}
        />
      </mesh>
      <lineSegments ref={wireRef} geometry={wireGeometry}>
        <lineBasicMaterial color="#4b5563" opacity={0.32} transparent />
      </lineSegments>
      {tracePoints.length > 1 ? (
        <Line
          points={tracePoints}
          vertexColors={traceColors}
          lineWidth={2}
          transparent
          opacity={0.95}
        />
      ) : null}
      <instancedMesh
        ref={instancedRef}
        args={[undefined, undefined, GRID_SIZE * GRID_SIZE]}
      >
        <sphereGeometry args={[0.08, 10, 10]} />
        <meshStandardMaterial
          color="#ffffff"
          emissive="#4b5563"
          emissiveIntensity={1.2}
          vertexColors
        />
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

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minPolarAngle={0.2}
        maxPolarAngle={1.2}
      />
    </>
  );
}

export default function Scene() {
  const { grid, stats, isRunning, setResult, setRunning, reset } =
    usePathStore();
  const [seed, setSeed] = useState(42);
  const [terrain, setTerrain] = useState(() => generateTerrain(seed));
  const [algorithm, setAlgorithm] = useState<Algorithm>("astar");
  const [heuristic, setHeuristic] = useState<Heuristic>("euclidean");
  const [speed, setSpeed] = useState(5);
  const [runError, setRunError] = useState<string | null>(null);
  const [solverTrace, setSolverTrace] = useState<TraceStep[]>([]);
  const [traceLines, setTraceLines] = useState<string[]>([]);
  const handleTraceStep = useCallback((line: string) => {
    setTraceLines((previous) => [line, ...previous].slice(0, 14));
  }, []);

  const fog = useMemo(() => {
    return new THREE.Fog("#f1efe8", 20, 80);
  }, []);

  const handleRun = async () => {
    if (!grid.length || isRunning) return;
    reset();
    setRunError(null);
    setSolverTrace([]);
    setTraceLines([]);
    setRunning(true);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/solve`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            grid,
            start: [0, 0],
            end: [GRID_SIZE - 1, GRID_SIZE - 1],
            algorithm,
            heuristic,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Solver request failed (${response.status})`);
      }

      const data = await response.json();
      setSolverTrace(Array.isArray(data.trace) ? data.trace : []);
      setResult({
        path: data.path || [],
        explored: data.explored || [],
        stats: data.stats,
      });
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
    const nextSeed = seed + 1;
    setSeed(nextSeed);
    setTerrain(generateTerrain(nextSeed));
  };

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
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
          onTraceStep={handleTraceStep}
        />
      </Canvas>

      <div
        style={{
          position: "absolute",
          top: 24,
          right: 24,
          zIndex: 10,
        }}
      >
        <ControlPanel
          algorithm={algorithm}
          heuristic={heuristic}
          speed={speed}
          onAlgorithmChange={setAlgorithm}
          onHeuristicChange={setHeuristic}
          onSpeedChange={setSpeed}
          onRegenerate={handleRegenerate}
          onRun={handleRun}
          stats={stats}
          isLoading={isRunning}
          error={runError}
          traceLines={traceLines}
        />
      </div>
    </div>
  );
}
