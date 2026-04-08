import { create } from "zustand";

type Stats = {
  nodes_explored: number;
  path_length: number;
  solve_time_ms: number;
};

type PathState = {
  grid: number[][];
  path: number[][];
  explored: number[][];
  isRunning: boolean;
  stats: Stats | null;
  summary: string;
  setResult: (data: {
    path: number[][];
    explored: number[][];
    stats: Stats;
    summary?: string;
  }) => void;
  setGrid: (grid: number[][]) => void;
  setRunning: (isRunning: boolean) => void;
  reset: () => void;
};

export const usePathStore = create<PathState>((set) => ({
  grid: [],
  path: [],
  explored: [],
  isRunning: false,
  stats: null,
  summary: "",
  setResult: ({ path, explored, stats, summary }) =>
    set({ path, explored, stats, summary: summary || "", isRunning: false }),
  setGrid: (grid) => set({ grid }),
  setRunning: (isRunning) => set({ isRunning }),
  reset: () =>
    set({ path: [], explored: [], stats: null, summary: "", isRunning: false }),
}));
