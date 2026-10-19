"""
optimizer.py -- Ant Colony Optimisation, Cuckoo Search and the Hybrid ACO-CSA.

PROBLEM DEFINITION
------------------
Given

  * a host with `total_cores` vCPU and `total_ram` MB of RAM,
  * N virtual machines, each with an observed worst-case demand
    (`req_cpu` cores, `req_ram` MB) taken from the simulated trace,

decide how much CPU and RAM to *assign* to every VM so that

  * no VM is starved below its peak demand  (starvation -> slowdown / crash),
  * no VM is grossly over-provisioned       (waste -> other VMs starve),
  * the sum of allocations never exceeds host capacity (hard constraint).

The scalar objective (lower is better) is

    J = w_short * unmetDemand
      + w_waste * wastedCapacity
      + w_risk  * shareOfVMsStillUnderProvisioned
      + w_viol  * hostCapacityOvershoot          (very heavily penalised)

ALGORITHMS
----------
ACO   : a construction graph of 12 discrete "allocation packages" per VM.
        Ants walk the VMs in random order and pick a package with
        probability proportional to pheromone^alpha * heuristic^beta, with a
        hard budget-feasibility bias so most colonies are already feasible.
        Evaporation + elitist deposit update the pheromone.

CSA   : continuous search. Nests hold real-valued (vCPU, MB) vectors.
        Leky-flighted candidates (Mantegna) plus a small best-guided pull are
        projected back into the feasible box + capacity budget, compared
        pairwise against a random nest, and the worst `pa` fraction is
        abandoned and re-seeded.

HYBRID: run ACO first (fast, discrete, good exploration), take its elite
        solutions, and use them to seed the CSA nest population which then
        refines them continuously.  Discrete polish -> continuous polish.

SAFETY: nothing here talks to VirtualBox. Outputs are pure numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from simulator import PRIORITY_WEIGHT, SimulationResult

__all__ = [
    "AllocationProblem",
    "OptimizationResult",
    "ACOOptimizer",
    "CSAOptimizer",
    "HybridACOCSA",
    "build_problem",
    "run_comparison",
]


# --------------------------------------------------------------------------- #
# Problem
# --------------------------------------------------------------------------- #
@dataclass
class AllocationProblem:
    """The resource-allocation search problem."""

    names: List[str]
    priorities: List[str]
    req_cpu: np.ndarray          # required vCPU per VM (observed peak)
    req_ram: np.ndarray          # required MB   per VM (observed peak)
    cur_cpu: np.ndarray          # current (naive) vCPU allocation
    cur_ram: np.ndarray          # current (naive) MB allocation
    total_cores: float
    total_ram: float

    # search box
    min_cpu: np.ndarray = field(default=None)
    max_cpu: np.ndarray = field(default=None)
    min_ram: np.ndarray = field(default=None)
    max_ram: np.ndarray = field(default=None)

    # discretisation used by ACO
    cpu_step: float = 0.5
    ram_step: float = 128.0
    n_levels: int = 12

    # objective weights
    w_short: float = 1.0
    w_waste: float = 0.15
    w_risk: float = 0.30
    w_violate: float = 15.0

    # filled in by __post_init__
    n: int = 0
    weights: np.ndarray = field(default=None, repr=False)
    cpu_levels: List[np.ndarray] = field(default=None, repr=False)
    ram_levels: List[np.ndarray] = field(default=None, repr=False)
    n_cpu: List[int] = field(default=None, repr=False)
    n_ram: List[int] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for attr in ("req_cpu", "req_ram", "cur_cpu", "cur_ram"):
            setattr(self, attr, np.asarray(getattr(self, attr), dtype=float))

        self.n = len(self.names)

        # ---- search box -------------------------------------------------- #
        if self.min_cpu is None:
            self.min_cpu = np.maximum(
                self.cpu_step,
                np.floor(0.50 * self.req_cpu / self.cpu_step) * self.cpu_step,
            )
        if self.max_cpu is None:
            self.max_cpu = np.ceil(1.20 * self.req_cpu / self.cpu_step) * self.cpu_step
        if self.min_ram is None:
            self.min_ram = np.maximum(
                self.ram_step,
                np.floor(0.60 * self.req_ram / self.ram_step) * self.ram_step,
            )
        if self.max_ram is None:
            self.max_ram = np.ceil(1.20 * self.req_ram / self.ram_step) * self.ram_step

        for attr in ("min_cpu", "max_cpu", "min_ram", "max_ram"):
            setattr(self, attr, np.asarray(getattr(self, attr), dtype=float))

        # Guard against a pathological box that already blows the budget.
        self.min_cpu = self._shrink_floor(self.min_cpu, self.total_cores)
        self.min_ram = self._shrink_floor(self.min_ram, self.total_ram)

        self.weights = np.array(
            [PRIORITY_WEIGHT.get(p, 2.0) for p in self.priorities], dtype=float
        )

        # ---- discrete candidate levels used by ACO ----------------------- #
        frac = np.linspace(0.0, 1.0, self.n_levels)
        self.cpu_levels, self.ram_levels = [], []
        self.n_cpu, self.n_ram = [], []
        for i in range(self.n):
            c = self.min_cpu[i] + frac * (self.max_cpu[i] - self.min_cpu[i])
            r = self.min_ram[i] + frac * (self.max_ram[i] - self.min_ram[i])
            c = np.unique(
                np.clip(
                    np.round(c / self.cpu_step) * self.cpu_step,
                    self.min_cpu[i],
                    self.max_cpu[i],
                )
            )
            r = np.unique(
                np.clip(
                    np.round(r / self.ram_step) * self.ram_step,
                    self.min_ram[i],
                    self.max_ram[i],
                )
            )
            self.cpu_levels.append(c.astype(float))
            self.ram_levels.append(r.astype(float))
            self.n_cpu.append(int(c.size))
            self.n_ram.append(int(r.size))

        self._w_req_cpu = float(np.sum(self.weights * self.req_cpu))
        self._w_req_ram = float(np.sum(self.weights * self.req_ram))
        self._w_total = float(np.sum(self.weights))

    # ---- helpers ---------------------------------------------------------- #
    def _shrink_floor(self, floor: np.ndarray, total: float) -> np.ndarray:
        """Make sure the lower bound itself fits inside the host budget."""
        floor = np.asarray(floor, dtype=float)
        if floor.sum() <= total:
            return floor
        return floor * (total / floor.sum())

    @staticmethod
    def _reduce(v: np.ndarray, total: float, floors: np.ndarray) -> np.ndarray:
        """Scale `v` down to `total` without going below `floors`."""
        s = float(v.sum())
        if s <= total + 1e-9:
            return v
        surplus = s - total
        head = float((v - floors).sum())
        if head <= 1e-12:
            return floors * (total / floors.sum())
        return np.maximum(floors, v - surplus * (v - floors) / head)

    def project(self, cpu: np.ndarray, ram: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clip into the box, then repair the capacity budget if needed."""
        cpu = np.clip(np.asarray(cpu, dtype=float), self.min_cpu, self.max_cpu)
        ram = np.clip(np.asarray(ram, dtype=float), self.min_ram, self.max_ram)
        cpu = self._reduce(cpu, self.total_cores, self.min_cpu)
        ram = self._reduce(ram, self.total_ram, self.min_ram)
        return cpu, ram

    def random_solution(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        cpu = rng.uniform(self.min_cpu, self.max_cpu)
        ram = rng.uniform(self.min_ram, self.max_ram)
        return self.project(cpu, ram)

    def levels_to_vec(
        self, idx_c: Sequence[int], idx_r: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        cpu = np.array(
            [self.cpu_levels[i][idx_c[i]] for i in range(self.n)], dtype=float
        )
        ram = np.array(
            [self.ram_levels[i][idx_r[i]] for i in range(self.n)], dtype=float
        )
        return cpu, ram

    def random_indices(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx_c = np.array(
            [int(rng.integers(0, self.n_cpu[i])) for i in range(self.n)], dtype=int
        )
        idx_r = np.array(
            [int(rng.integers(0, self.n_ram[i])) for i in range(self.n)], dtype=int
        )
        return idx_c, idx_r

    # ---- intensification / local search ---------------------------------- #
    def grid_polish(
        self,
        idx_c: np.ndarray,
        idx_r: np.ndarray,
        max_moves: int = 600,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Steepest descent on the *discrete* grid using 1-to-1 transfers.

        One VM is upgraded while another is downgraded, so the host budget is
        preserved exactly. CPU and RAM move independently.

        Returns ``(idx_c, idx_r, evaluations)``.
        """
        idx_c = idx_c.copy()
        idx_r = idx_r.copy()
        evals = 0

        for _ in range(max_moves):
            cpu, ram = self.levels_to_vec(idx_c, idx_r)
            best_f = self.fitness(cpu, ram)
            evals += 1
            best_move: Optional[Tuple[np.ndarray, np.ndarray]] = None

            for i in range(self.n):
                if idx_c[i] + 1 >= self.n_cpu[i]:
                    continue
                for j in range(self.n):
                    if i == j or idx_c[j] - 1 < 0:
                        continue
                    tc = idx_c.copy()
                    tc[i] += 1
                    tc[j] -= 1
                    c2, r2 = self.levels_to_vec(tc, idx_r)
                    if c2.sum() > self.total_cores + 1e-9:
                        continue
                    f = self.fitness(c2, r2)
                    evals += 1
                    if f < best_f - 1e-12:
                        best_f = f
                        best_move = (tc, idx_r.copy())

            for i in range(self.n):
                if idx_r[i] + 1 >= self.n_ram[i]:
                    continue
                for j in range(self.n):
                    if i == j or idx_r[j] - 1 < 0:
                        continue
                    tr = idx_r.copy()
                    tr[i] += 1
                    tr[j] -= 1
                    c2, r2 = self.levels_to_vec(idx_c, tr)
                    if r2.sum() > self.total_ram + 1e-9:
                        continue
                    f = self.fitness(c2, r2)
                    evals += 1
                    if f < best_f - 1e-12:
                        best_f = f
                        best_move = (idx_c.copy(), tr)

            if best_move is None:
                break
            idx_c, idx_r = best_move

        return idx_c, idx_r, evals

    def coordinate_polish(
        self,
        cpu: np.ndarray,
        ram: np.ndarray,
        cpu_steps: Sequence[float] = (0.25, 0.10, 0.05, 0.02),
        ram_steps: Sequence[float] = (64.0, 32.0, 16.0, 8.0),
        max_rounds: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Greedy coordinate descent on the continuous allocation.

        Returns ``(cpu, ram, fitness, evaluations)``.
        """
        cpu, ram = self.project(cpu, ram)
        best_f = self.fitness(cpu, ram)
        evals = 1
        for _ in range(max_rounds):
            improved = False
            for i in range(self.n):
                for step in cpu_steps:
                    for delta in (step, -step):
                        trial = cpu.copy()
                        trial[i] += delta
                        tc, tr = self.project(trial, ram)
                        f = self.fitness(tc, tr)
                        evals += 1
                        if f < best_f - 1e-12:
                            cpu, ram, best_f = tc, tr, f
                            improved = True
                for step in ram_steps:
                    for delta in (step, -step):
                        trial = ram.copy()
                        trial[i] += delta
                        tc, tr = self.project(cpu, trial)
                        f = self.fitness(tc, tr)
                        evals += 1
                        if f < best_f - 1e-12:
                            cpu, ram, best_f = tc, tr, f
                            improved = True
            if not improved:
                break
        return cpu, ram, best_f, evals

    # ---- objective -------------------------------------------------------- #
    def fitness(self, cpu: np.ndarray, ram: np.ndarray) -> float:
        cpu = np.asarray(cpu, dtype=float)
        ram = np.asarray(ram, dtype=float)

        # 1. unmet demand -> slowdown / crash risk (priority weighted)
        cpu_short = np.maximum(0.0, self.req_cpu - cpu)
        ram_short = np.maximum(0.0, self.req_ram - ram)
        short = (
            np.sum(self.weights * cpu_short) / self._w_req_cpu
            + np.sum(self.weights * ram_short) / self._w_req_ram
        ) / 2.0

        # 2. provisioned-but-never-needed capacity (efficiency)
        cpu_waste = np.maximum(0.0, cpu - self.req_cpu)
        ram_waste = np.maximum(0.0, ram - self.req_ram)
        waste = (
            cpu_waste.sum() / self.total_cores + ram_waste.sum() / self.total_ram
        ) / 2.0

        # 3. hard capacity overshoot (must stay ~0)
        viol = (
            max(0.0, float(cpu.sum()) - self.total_cores) / self.total_cores
            + max(0.0, float(ram.sum()) - self.total_ram) / self.total_ram
        )

        # 4. share of VMs still under-provisioned (coarse risk signal)
        risk = (
            np.sum(self.weights * (cpu < self.req_cpu)) / self._w_total
            + np.sum(self.weights * (ram < self.req_ram)) / self._w_total
        ) / 2.0

        return float(
            self.w_short * short
            + self.w_waste * waste
            + self.w_risk * risk
            + self.w_violate * viol
        )

    def local_eta(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Pre-computed per-VM attractiveness of every CPU / RAM level."""
        eta_c: List[np.ndarray] = []
        eta_r: List[np.ndarray] = []
        for i in range(self.n):
            ec = np.empty(self.n_cpu[i], dtype=float)
            for k in range(self.n_cpu[i]):
                c = float(self.cpu_levels[i][k])
                short = max(0.0, self.req_cpu[i] - c) / max(self.req_cpu[i], 1e-9)
                waste = max(0.0, c - self.req_cpu[i]) / self.total_cores
                ec[k] = 1.0 / (self.w_short * short + self.w_waste * waste + 1e-6)

            er = np.empty(self.n_ram[i], dtype=float)
            for m in range(self.n_ram[i]):
                r = float(self.ram_levels[i][m])
                short = max(0.0, self.req_ram[i] - r) / max(self.req_ram[i], 1e-9)
                waste = max(0.0, r - self.req_ram[i]) / self.total_ram
                er[m] = 1.0 / (self.w_short * short + self.w_waste * waste + 1e-6)

            eta_c.append(ec)
            eta_r.append(er)
        return eta_c, eta_r

    # ---- reporting -------------------------------------------------------- #
    def summary(self, cpu: np.ndarray, ram: np.ndarray) -> Dict[str, float]:
        cpu = np.asarray(cpu, dtype=float)
        ram = np.asarray(ram, dtype=float)
        cpu_waste = np.maximum(0.0, cpu - self.req_cpu)
        ram_waste = np.maximum(0.0, ram - self.req_ram)
        cpu_short = np.maximum(0.0, self.req_cpu - cpu)
        ram_short = np.maximum(0.0, self.req_ram - ram)
        return {
            "fitness": self.fitness(cpu, ram),
            "cpu_allocated": float(cpu.sum()),
            "ram_allocated": float(ram.sum()),
            "cpu_wasted": float(cpu_waste.sum()),
            "ram_wasted": float(ram_waste.sum()),
            "cpu_unmet": float(cpu_short.sum()),
            "ram_unmet": float(ram_short.sum()),
            "under_provisioned": int(
                np.sum((cpu < self.req_cpu - 1e-9) | (ram < self.req_ram - 1e-9))
            ),
            "cpu_ok": bool(cpu.sum() <= self.total_cores + 1e-9),
            "ram_ok": bool(ram.sum() <= self.total_ram + 1e-9),
        }


def build_problem(sim: SimulationResult) -> AllocationProblem:
    """Turn a :class:`SimulationResult` into an :class:`AllocationProblem`."""
    return AllocationProblem(
        names=sim.names,
        priorities=sim.priorities,
        req_cpu=sim.need_cpu,
        req_ram=sim.need_ram,
        cur_cpu=sim.current_cpu,
        cur_ram=sim.current_ram,
        total_cores=float(sim.host.total_cores),
        total_ram=float(sim.host.total_ram),
    )


# --------------------------------------------------------------------------- #
# Shared result container
# --------------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    method: str
    best_cpu: np.ndarray
    best_ram: np.ndarray
    best_fitness: float
    history: List[float]
    evaluations: int
    elapsed: float
    elite: List[Tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, float]:
        return {"fitness": self.best_fitness, "evals": self.evaluations,
                "seconds": self.elapsed}


# --------------------------------------------------------------------------- #
# Ant Colony Optimisation
# --------------------------------------------------------------------------- #
class ACOOptimizer:
    """Discrete ACO over per-VM allocation packages."""

    def __init__(
        self,
        n_ants: int = 22,
        iterations: int = 60,
        alpha: float = 1.0,
        beta: float = 2.5,
        rho: float = 0.25,
        q: float = 1.0,
        tau_max: float = 25.0,
        seed: int = 42,
    ) -> None:
        self.n_ants = int(n_ants)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.tau_max = float(tau_max)
        self.seed = int(seed)

    # -- construction ------------------------------------------------------- #
    def _construct(
        self,
        problem: AllocationProblem,
        tau_c: List[np.ndarray],
        tau_r: List[np.ndarray],
        eta_c: List[np.ndarray],
        eta_r: List[np.ndarray],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = problem.n
        idx_c = np.zeros(n, dtype=int)
        idx_r = np.zeros(n, dtype=int)
        cpu_used = 0.0
        ram_used = 0.0

        for i in rng.permutation(n):
            # --- CPU level -------------------------------------------------- #
            w = (tau_c[i] ** self.alpha) * (eta_c[i] ** self.beta)
            room = problem.total_cores - cpu_used
            ok = problem.cpu_levels[i] <= room + 1e-9
            w = np.where(ok, w, w.max() * 1e-6 + 1e-12)
            p = w / float(w.sum())
            k = int(rng.choice(w.size, p=p))
            idx_c[i] = k
            cpu_used += float(problem.cpu_levels[i][k])

            # --- RAM level -------------------------------------------------- #
            w = (tau_r[i] ** self.alpha) * (eta_r[i] ** self.beta)
            room = problem.total_ram - ram_used
            ok = problem.ram_levels[i] <= room + 1e-9
            w = np.where(ok, w, w.max() * 1e-6 + 1e-12)
            p = w / float(w.sum())
            m = int(rng.choice(w.size, p=p))
            idx_r[i] = m
            ram_used += float(problem.ram_levels[i][m])

        return self._repair(problem, idx_c, idx_r)

    def _repair(
        self, problem: AllocationProblem, idx_c: np.ndarray, idx_r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Force a construction back inside the host budget by downgrading."""
        idx_c = idx_c.copy()
        idx_r = idx_r.copy()

        for _ in range(600):
            cpu, ram = problem.levels_to_vec(idx_c, idx_r)
            over_c = float(cpu.sum()) - problem.total_cores
            over_r = float(ram.sum()) - problem.total_ram
            if over_c <= 1e-9 and over_r <= 1e-9:
                break

            moved = False
            if over_c > 1e-9:
                cand = [i for i in range(problem.n) if idx_c[i] > 0]
                if cand:
                    i = max(cand, key=lambda j: cpu[j] / max(problem.req_cpu[j], 1e-9))
                    idx_c[i] -= 1
                    moved = True
            if over_r > 1e-9:
                cand = [i for i in range(problem.n) if idx_r[i] > 0]
                if cand:
                    i = max(cand, key=lambda j: ram[j] / max(problem.req_ram[j], 1e-9))
                    idx_r[i] -= 1
                    moved = True
            if not moved:
                break

        # Ultimate fallback: the all-minimum assignment is always feasible.
        cpu, ram = problem.levels_to_vec(idx_c, idx_r)
        if (
            cpu.sum() > problem.total_cores + 1e-9
            or ram.sum() > problem.total_ram + 1e-9
        ):  # pragma: no cover - defensive
            idx_c = np.zeros(problem.n, dtype=int)
            idx_r = np.zeros(problem.n, dtype=int)
        return idx_c, idx_r

    # -- main loop ---------------------------------------------------------- #
    def optimize(
        self, problem: AllocationProblem, rng: Optional[np.random.Generator] = None
    ) -> OptimizationResult:
        rng = rng or np.random.default_rng(self.seed)
        t0 = time.perf_counter()
        n = problem.n

        tau_c = [np.ones(problem.n_cpu[i], dtype=float) for i in range(n)]
        tau_r = [np.ones(problem.n_ram[i], dtype=float) for i in range(n)]
        eta_c, eta_r = problem.local_eta()

        # Scale factor so pheromone deposits stay in a sane range.
        probe = [problem.fitness(*problem.random_solution(rng)) for _ in range(12)]
        f_scale = max(1e-6, float(np.median(probe)))
        evals = len(probe)

        history: List[float] = []
        best_f = float("inf")
        best_idx: Optional[Tuple[np.ndarray, np.ndarray]] = None
        archive: List[Tuple[float, np.ndarray, np.ndarray]] = []

        for _ in range(self.iterations):
            ants: List[Tuple[np.ndarray, np.ndarray, float]] = []
            for _ in range(self.n_ants):
                ic, ir = self._construct(problem, tau_c, tau_r, eta_c, eta_r, rng)
                f = problem.fitness(*problem.levels_to_vec(ic, ir))
                evals += 1
                ants.append((ic, ir, f))
                archive.append((f, *problem.levels_to_vec(ic, ir)))

            ants.sort(key=lambda a: a[2])

            # evaporate
            tau_c = [np.maximum((1.0 - self.rho) * t, 1e-3) for t in tau_c]
            tau_r = [np.maximum((1.0 - self.rho) * t, 1e-3) for t in tau_r]

            def deposit(ic: np.ndarray, ir: np.ndarray, f: float) -> None:
                d = min(self.q / (f + f_scale), self.tau_max)
                for i in range(n):
                    tau_c[i][int(ic[i])] = min(tau_c[i][int(ic[i])] + d, self.tau_max)
                    tau_r[i][int(ir[i])] = min(tau_r[i][int(ir[i])] + d, self.tau_max)

            # iteration-best
            deposit(ants[0][0], ants[0][1], ants[0][2])

            # global best + elitist reinforcement
            if ants[0][2] < best_f:
                best_f = ants[0][2]
                best_idx = (ants[0][0].copy(), ants[0][1].copy())
            if best_idx is not None:
                deposit(best_idx[0], best_idx[1], best_f)

            history.append(float(best_f))

        if best_idx is None:                       # pragma: no cover - safety
            best_idx = (np.zeros(n, dtype=int), np.zeros(n, dtype=int))

        # Intensification: steepest descent over discrete transfer moves.
        idx_c, idx_r, polish_evals = problem.grid_polish(*best_idx)
        evals += polish_evals
        best_cpu, best_ram = problem.levels_to_vec(idx_c, idx_r)
        best_f = problem.fitness(best_cpu, best_ram)
        if history:
            history[-1] = min(history[-1], float(best_f))

        archive.sort(key=lambda a: a[0])
        elite: List[Tuple[np.ndarray, np.ndarray, float]] = []
        seen = set()
        for f, c, r in archive:
            key = tuple(np.round(np.concatenate([c, r]), 3))
            if key in seen:
                continue
            seen.add(key)
            elite.append((c, r, float(f)))
            if len(elite) >= 8:
                break

        return OptimizationResult(
            method="ACO",
            best_cpu=best_cpu,
            best_ram=best_ram,
            best_fitness=float(best_f),
            history=history,
            evaluations=evals,
            elapsed=time.perf_counter() - t0,
            elite=elite,
        )


# --------------------------------------------------------------------------- #
# Cuckoo Search
# --------------------------------------------------------------------------- #
def _levy(size: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Mantegna's algorithm for a stable distribution with index `lam`."""
    u = rng.normal(0.0, 1.0, size)
    v = rng.normal(0.0, 1.0, size)
    return u / (np.abs(v) ** (1.0 / lam) + 1e-12)


class CSAOptimizer:
    """Continuous Cuckoo Search with Leky flights + a best-guided pull."""

    def __init__(
        self,
        n_nests: int = 24,
        generations: int = 220,
        pa: float = 0.25,
        alpha: float = 0.04,
        guide: float = 0.30,
        lam: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.n_nests = int(n_nests)
        self.generations = int(generations)
        self.pa = float(pa)
        self.alpha = float(alpha)
        self.guide = float(guide)
        self.lam = float(lam)
        self.seed = int(seed)

    def optimize(
        self,
        problem: AllocationProblem,
        initial: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> OptimizationResult:
        rng = rng or np.random.default_rng(self.seed)
        t0 = time.perf_counter()

        span_cpu = problem.max_cpu - problem.min_cpu
        span_ram = problem.max_ram - problem.min_ram
        dim = problem.n * 2
        evals = 0

        # ---- seed the population ----------------------------------------- #
        nests: List[np.ndarray] = []
        if initial:
            for c, r in initial[: self.n_nests]:
                c2, r2 = problem.project(c, r)
                nests.append(np.concatenate([c2, r2]))

        while len(nests) < self.n_nests:
            c, r = problem.random_solution(rng)
            nests.append(np.concatenate([c, r]))

        fits = [problem.fitness(x[: problem.n], x[problem.n:]) for x in nests]
        evals += len(fits)

        best_i = int(np.argmin(fits))
        history: List[float] = [float(fits[best_i])]

        for g in range(self.generations):
            # Anneal the Leky step so the search contracts over time.
            a_t = self.alpha * (1.0 - 0.6 * g / max(1, self.generations - 1))
            best = nests[best_i]

            for i in range(self.n_nests):
                step = _levy(dim, self.lam, rng)
                span = np.concatenate([span_cpu, span_ram])
                cand = (
                    nests[i]
                    + a_t * span * step
                    + self.guide * (best - nests[i])
                )
                cand = np.concatenate(problem.project(cand[: problem.n], cand[problem.n:]))
                f = problem.fitness(cand[: problem.n], cand[problem.n:])
                evals += 1

                # Standard CSA: compare against a randomly drawn nest.
                j = int(rng.integers(0, self.n_nests))
                if f < fits[j]:
                    nests[j], fits[j] = cand, f

            # Abandon the worst fraction and re-seed (idiotype refresh).
            n_abandon = max(1, int(self.pa * self.n_nests))
            worst = np.argsort(fits)[-n_abandon:]
            for w in worst:
                c, r = problem.random_solution(rng)
                nests[w] = np.concatenate([c, r])
                fits[w] = problem.fitness(c, r)
                evals += 1

            best_i = int(np.argmin(fits))
            history.append(float(fits[best_i]))

        # Intensification: greedy coordinate descent on the continuous box.
        b_cpu = nests[best_i][: problem.n].copy()
        b_ram = nests[best_i][problem.n:].copy()
        b_cpu, b_ram, b_f, polish_evals = problem.coordinate_polish(b_cpu, b_ram)
        evals += polish_evals
        if b_f < fits[best_i]:
            fits[best_i] = b_f
            nests[best_i] = np.concatenate([b_cpu, b_ram])
        history[-1] = min(history[-1], float(b_f))

        order = np.argsort(fits)
        elite = [
            (
                nests[k][: problem.n].copy(),
                nests[k][problem.n:].copy(),
                float(fits[k]),
            )
            for k in order[:8]
        ]

        return OptimizationResult(
            method="CSA",
            best_cpu=nests[best_i][: problem.n].copy(),
            best_ram=nests[best_i][problem.n:].copy(),
            best_fitness=float(fits[best_i]),
            history=history,
            evaluations=evals,
            elapsed=time.perf_counter() - t0,
            elite=elite,
        )


# --------------------------------------------------------------------------- #
# Hybrid ACO -> CSA
# --------------------------------------------------------------------------- #
class HybridACOCSA:
    """ACO explores the discrete grid, CSA polishes the result continuously."""

    def __init__(
        self,
        aco_ants: int = 20,
        aco_iterations: int = 40,
        csa_nests: int = 24,
        csa_generations: int = 160,
        elite_n: int = 6,
        seed: int = 42,
    ) -> None:
        self.aco = ACOOptimizer(n_ants=aco_ants, iterations=aco_iterations, seed=seed)
        self.csa = CSAOptimizer(n_nests=csa_nests, generations=csa_generations, seed=seed + 1)
        self.elite_n = int(elite_n)
        self.seed = int(seed)

    def optimize(
        self, problem: AllocationProblem, rng: Optional[np.random.Generator] = None
    ) -> OptimizationResult:
        rng = rng or np.random.default_rng(self.seed)
        t0 = time.perf_counter()

        # Phase 1 - discrete exploration
        aco_res = self.aco.optimize(problem, rng)

        # Phase 2 - continuous refinement seeded with ACO's elites
        seed_nests = [(c, r) for c, r, _ in aco_res.elite[: self.elite_n]]
        csa_res = self.csa.optimize(problem, initial=seed_nests, rng=rng)

        if csa_res.best_fitness <= aco_res.best_fitness:
            best_cpu, best_ram, best_f = csa_res.best_cpu, csa_res.best_ram, csa_res.best_fitness
        else:                                       # pragma: no cover - rare
            best_cpu, best_ram, best_f = aco_res.best_cpu, aco_res.best_ram, aco_res.best_fitness

        history = list(aco_res.history) + list(csa_res.history)
        elite = sorted(
            aco_res.elite + csa_res.elite, key=lambda e: e[2]
        )[:8]

        return OptimizationResult(
            method="Hybrid ACO-CSA",
            best_cpu=best_cpu,
            best_ram=best_ram,
            best_fitness=float(best_f),
            history=history,
            evaluations=aco_res.evaluations + csa_res.evaluations,
            elapsed=time.perf_counter() - t0,
            elite=elite,
        )


# --------------------------------------------------------------------------- #
# Convenience: run all three on the same problem
# --------------------------------------------------------------------------- #
def run_comparison(
    problem: AllocationProblem,
    seed: int = 42,
    aco_ants: int = 20,
    aco_iterations: int = 40,
    csa_nests: int = 24,
    csa_generations: int = 160,
) -> Dict[str, OptimizationResult]:
    """Run ACO, CSA and the Hybrid on one problem and return all three."""
    results: Dict[str, OptimizationResult] = {}

    aco = ACOOptimizer(n_ants=aco_ants, iterations=aco_iterations, seed=seed)
    results["ACO"] = aco.optimize(problem, np.random.default_rng(seed))

    csa = CSAOptimizer(n_nests=csa_nests, generations=csa_generations, seed=seed)
    results["CSA"] = csa.optimize(problem, rng=np.random.default_rng(seed))

    hyb = HybridACOCSA(
        aco_ants=aco_ants,
        aco_iterations=aco_iterations,
        csa_nests=csa_nests,
        csa_generations=csa_generations,
        seed=seed,
    )
    results["Hybrid ACO-CSA"] = hyb.optimize(problem, np.random.default_rng(seed))

    return results


# --------------------------------------------------------------------------- #
# Self-test when run directly:  python optimizer.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from simulator import simulate

    sim = simulate(days=7, seed=42)
    problem = build_problem(sim)

    print(f"Problem: {problem.n} VMs, {problem.total_cores:g} vCPU, "
          f"{problem.total_ram} MB")
    print(f"Required  CPU {problem.req_cpu.sum():6.2f} cores "
          f"| RAM {problem.req_ram.sum():7.0f} MB")
    print(f"Minimum   CPU {problem.min_cpu.sum():6.2f} cores "
          f"| RAM {problem.min_ram.sum():7.0f} MB")
    print(f"CPU levels per VM: {problem.n_cpu}")
    print(f"RAM levels per VM: {problem.n_ram}")

    print(f"\nBEFORE  fitness = {problem.fitness(problem.cur_cpu, problem.cur_ram):.5f}"
          f"  {problem.summary(problem.cur_cpu, problem.cur_ram)}")

    for name, res in run_comparison(problem, seed=42).items():
        s = problem.summary(res.best_cpu, res.best_ram)
        print(f"\n{name:<16} fitness={res.best_fitness:.5f}  "
              f"evals={res.evaluations:<5d} {res.elapsed:.2f}s")
        print(f"   cpu = {np.round(res.best_cpu, 2)}")
        print(f"   ram = {np.round(res.best_ram, 0)}")
        print(f"   {s}")
