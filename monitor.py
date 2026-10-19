"""
monitor.py -- Read-only resource monitoring and overload-risk detection.

Two jobs:

1. :class:`HostMonitor` - a *read-only* wrapper around ``psutil`` that reports
   what the real Windows machine is doing right now (CPU %, RAM, load, top
   processes). It only ever reads. It never changes VirtualBox settings and
   never stops or restarts a VM.

2. :class:`RiskDetector` - takes a simulated demand trace plus a proposed
   CPU/RAM allocation and answers the question this whole project is about:

       "Which VMs are in danger of being starved, and how bad is it?"

   The same detector is used for the "before" (current VirtualBox) allocation
   and the "after" (optimised) allocation, so the two numbers are directly
   comparable.

SAFETY: nothing in this module writes to the system or to VirtualBox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from simulator import HostSpec, SimulationResult

try:                                    # psutil is optional at import time
    import psutil
except Exception:                       # pragma: no cover - environment specific
    psutil = None

__all__ = [
    "HostSnapshot",
    "VMRisk",
    "HostRisk",
    "RiskReport",
    "HostMonitor",
    "RiskDetector",
    "detect_overload",
    "LEVEL_ORDER",
    "LEVEL_EMOJI",
]


LEVEL_ORDER: List[str] = ["critical", "high", "medium", "low", "ok"]
LEVEL_EMOJI: Dict[str, str] = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "ok": "🟢",
    "waste": "⚠️",
}


# =========================================================================== #
# 1. Live host monitoring (read-only, via psutil)
# =========================================================================== #
@dataclass
class HostSnapshot:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    ram_total_mb: float
    core_count: int
    logical_cpus: int
    load_per_core: float
    top_processes: List[Tuple[str, float, float]] = field(default_factory=list)

    @property
    def level(self) -> str:
        worst = max(self.cpu_percent, self.ram_percent)
        if worst >= 90:
            return "critical"
        if worst >= 80:
            return "high"
        if worst >= 65:
            return "medium"
        if worst >= 45:
            return "low"
        return "ok"


class HostMonitor:
    """Read-only sampler for the physical machine running the dashboard."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = float(interval)
        if psutil is not None:
            psutil.cpu_percent(interval=None)      # prime the internal counter

    # -- capability --------------------------------------------------------- #
    @property
    def available(self) -> bool:
        return psutil is not None

    def unavailable_reason(self) -> str:
        return (
            "psutil is not installed - run `pip install psutil` to enable live "
            "host monitoring. Everything else in the project still works, "
            "because the optimiser runs on simulated data."
        )

    # -- sampling ----------------------------------------------------------- #
    def snapshot(self, top_n: int = 5) -> HostSnapshot:
        import time as _time

        if psutil is None:                         # graceful degradation
            return HostSnapshot(
                timestamp=_time.time(),
                cpu_percent=0.0,
                ram_percent=0.0,
                ram_used_mb=0.0,
                ram_total_mb=0.0,
                core_count=0,
                logical_cpus=0,
                load_per_core=0.0,
                top_processes=[],
            )

        cpu = float(psutil.cpu_percent(interval=self.interval))
        vm = psutil.virtual_memory()
        logical = int(psutil.cpu_count(logical=True) or 1)
        physical = int(psutil.cpu_count(logical=False) or logical)

        try:
            load1 = float(psutil.getloadavg()[0])
        except (AttributeError, OSError):          # Windows has no getloadavg
            load1 = cpu / 100.0 * logical
        load_per_core = load1 / max(1, logical)

        procs: List[Tuple[str, float, float]] = []
        try:
            rows = []
            for p in psutil.process_iter(["name", "cpu_percent", "memory_info"]):
                info = p.info
                mem = info.get("memory_info")
                mem_mb = float(getattr(mem, "rss", 0)) / (1024 * 1024)
                rows.append((str(info.get("name") or "?"),
                             float(info.get("cpu_percent") or 0.0), mem_mb))
            rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
            procs = rows[:top_n]
        except Exception:                          # pragma: no cover
            procs = []

        return HostSnapshot(
            timestamp=_time.time(),
            cpu_percent=cpu,
            ram_percent=float(vm.percent),
            ram_used_mb=float(vm.used) / (1024 * 1024),
            ram_total_mb=float(vm.total) / (1024 * 1024),
            core_count=physical,
            logical_cpus=logical,
            load_per_core=load_per_core,
            top_processes=procs,
        )

    def sample(self, count: int = 5) -> List[HostSnapshot]:
        return [self.snapshot() for _ in range(max(1, int(count)))]


# =========================================================================== #
# 2. Overload-risk model
# =========================================================================== #
@dataclass
class VMRisk:
    name: str
    priority: str
    alloc_cpu: float
    alloc_ram: float
    need_cpu: float
    need_ram: float
    cpu_util_peak: float
    ram_util_peak: float
    cpu_over_hours: int
    ram_over_hours: int
    over_hours: int
    satisfaction: float
    wasted_cpu: float
    wasted_ram: float
    level: str
    reasons: List[str] = field(default_factory=list)

    @property
    def cpu_short(self) -> float:
        return max(0.0, self.need_cpu - self.alloc_cpu)

    @property
    def ram_short(self) -> float:
        return max(0.0, self.need_ram - self.alloc_ram)

    @property
    def waste_ratio(self) -> float:
        if self.alloc_cpu <= 0:
            return 0.0
        return self.wasted_cpu / self.alloc_cpu


@dataclass
class HostRisk:
    cpu_alloc: float
    ram_alloc: float
    cpu_total: float
    ram_total: float
    peak_demand_cpu: float
    peak_demand_ram: float
    cpu_alarm: float
    ram_alarm: float
    level: str
    reasons: List[str] = field(default_factory=list)

    @property
    def cpu_headroom(self) -> float:
        return self.cpu_total - self.cpu_alloc

    @property
    def ram_headroom(self) -> float:
        return self.ram_total - self.ram_alloc

    @property
    def cpu_alloc_pct(self) -> float:
        return self.cpu_alloc / self.cpu_total if self.cpu_total else 0.0

    @property
    def ram_alloc_pct(self) -> float:
        return self.ram_alloc / self.ram_total if self.ram_total else 0.0

    @property
    def peak_demand_cpu_pct(self) -> float:
        return self.peak_demand_cpu / self.cpu_total if self.cpu_total else 0.0

    @property
    def peak_demand_ram_pct(self) -> float:
        return self.peak_demand_ram / self.ram_total if self.ram_total else 0.0


@dataclass
class RiskReport:
    label: str
    vm_risks: List[VMRisk]
    host: HostRisk
    score: float
    level: str
    counts: Dict[str, int]
    overloaded_vm_hours: int
    total_vm_hours: int
    wasted_cpu: float
    wasted_ram: float
    unmet_cpu: float
    unmet_ram: float
    mean_satisfaction: float

    @property
    def overloaded_pct(self) -> float:
        return (
            self.overloaded_vm_hours / self.total_vm_hours
            if self.total_vm_hours
            else 0.0
        )

    @property
    def under_provisioned(self) -> List[VMRisk]:
        return [v for v in self.vm_risks if v.level in ("critical", "high")]

    def by_name(self) -> Dict[str, VMRisk]:
        return {v.name: v for v in self.vm_risks}


def _level_from_score(score: float) -> str:
    if score >= 60:
        return "critical"
    if score >= 35:
        return "high"
    if score >= 20:
        return "medium"
    if score >= 8:
        return "low"
    return "ok"


class RiskDetector:
    """Replays a demand trace against an allocation and grades the risk."""

    def __init__(
        self,
        stressed: float = 0.90,
        overload: float = 1.00,
        host_stressed: float = 0.85,
    ) -> None:
        self.stressed = float(stressed)
        self.overload = float(overload)
        self.host_stressed = float(host_stressed)

    # -- per VM ------------------------------------------------------------- #
    def _vm_risk(
        self,
        name: str,
        priority: str,
        weight: float,
        demand_cpu: np.ndarray,
        demand_ram: np.ndarray,
        alloc_cpu: float,
        alloc_ram: float,
        need_cpu: float,
        need_ram: float,
    ) -> VMRisk:
        cpu_u = demand_cpu / max(alloc_cpu, 1e-9)
        ram_u = demand_ram / max(alloc_ram, 1e-9)

        cpu_over = int(np.count_nonzero(cpu_u > self.overload))
        ram_over = int(np.count_nonzero(ram_u > self.overload))
        over_hours = int(np.count_nonzero((cpu_u > self.overload) | (ram_u > self.overload)))

        sat_cpu = float(np.minimum(1.0, alloc_cpu / np.maximum(demand_cpu, 1e-9)).mean())
        sat_ram = float(np.minimum(1.0, alloc_ram / np.maximum(demand_ram, 1e-9)).mean())
        satisfaction = 0.5 * (sat_cpu + sat_ram)

        wasted_cpu = max(0.0, alloc_cpu - need_cpu)
        wasted_ram = max(0.0, alloc_ram - need_ram)

        cpu_peak = float(cpu_u.max())
        ram_peak = float(ram_u.max())

        reasons: List[str] = []
        if cpu_peak > self.overload:
            reasons.append(
                f"CPU demand exceeds the assigned {alloc_cpu:.2f} vCPU for "
                f"{cpu_over} of {demand_cpu.size} hours (peak {cpu_peak * 100:.0f} % "
                f"of allocation)."
            )
        elif cpu_peak > self.stressed:
            reasons.append(
                f"CPU peaks at {cpu_peak * 100:.0f} % of the {alloc_cpu:.2f} vCPU "
                f"assigned. The allocation is sized to the measured peak, so there "
                f"is little burst headroom left - acceptable, but worth watching."
            )
        if ram_peak > self.overload:
            reasons.append(
                f"RAM demand exceeds the assigned {alloc_ram:.0f} MB for "
                f"{ram_over} of {demand_ram.size} hours (peak {ram_peak * 100:.0f} % "
                f"of allocation) - the guest may start swapping."
            )
        elif ram_peak > self.stressed:
            reasons.append(
                f"RAM runs close to the limit (peak {ram_peak * 100:.0f} % of the "
                f"assigned {alloc_ram:.0f} MB)."
            )
        if wasted_cpu > 0.75:
            reasons.append(
                f"Over-provisioned: {wasted_cpu:.2f} vCPU is assigned but never "
                f"used (peak demand is only {need_cpu:.2f} vCPU)."
            )
        if wasted_ram > 512:
            reasons.append(
                f"Over-provisioned: {wasted_ram:.0f} MB is assigned but never used "
                f"(peak demand is only {need_ram:.0f} MB)."
            )
        if not reasons:
            reasons.append(
                f"Healthy: peak CPU {cpu_peak * 100:.0f} % and peak RAM "
                f"{ram_peak * 100:.0f} % of the assigned resources."
            )

        # ---- severity ----------------------------------------------------- #
        # Severity is driven by *starvation* (demand > allocation), not by the
        # plain utilisation of the allocation: a VM sized exactly to its
        # measured peak legitimately runs at ~100 % at that peak.
        horizon = max(1, int(demand_cpu.size))
        heavy = over_hours >= max(3, int(0.05 * horizon))
        if heavy or cpu_peak > 1.15 or ram_peak > 1.15:
            level = "critical"
        elif over_hours > 0:
            level = "high"
        elif (alloc_cpu > 0 and wasted_cpu / max(alloc_cpu, 1e-9) > 0.35) or (
            alloc_ram > 0 and wasted_ram / max(alloc_ram, 1e-9) > 0.35
        ):
            level = "waste"
        else:
            level = "ok"

        return VMRisk(
            name=name,
            priority=priority,
            alloc_cpu=float(alloc_cpu),
            alloc_ram=float(alloc_ram),
            need_cpu=float(need_cpu),
            need_ram=float(need_ram),
            cpu_util_peak=cpu_peak,
            ram_util_peak=ram_peak,
            cpu_over_hours=cpu_over,
            ram_over_hours=ram_over,
            over_hours=over_hours,
            satisfaction=satisfaction,
            wasted_cpu=wasted_cpu,
            wasted_ram=wasted_ram,
            level=level,
            reasons=reasons,
        )

    # -- whole fleet -------------------------------------------------------- #
    def analyse(
        self,
        sim: SimulationResult,
        cpu_alloc: Sequence[float],
        ram_alloc: Sequence[float],
        label: str = "allocation",
    ) -> RiskReport:
        cpu_alloc = np.asarray(cpu_alloc, dtype=float)
        ram_alloc = np.asarray(ram_alloc, dtype=float)

        vm_risks: List[VMRisk] = []
        weights: List[float] = []
        for i, profile in enumerate(sim.profiles):
            weights.append(profile.weight)
            vm_risks.append(
                self._vm_risk(
                    name=profile.name,
                    priority=profile.priority,
                    weight=profile.weight,
                    demand_cpu=sim.cpu[i],
                    demand_ram=sim.ram[i],
                    alloc_cpu=float(cpu_alloc[i]),
                    alloc_ram=float(ram_alloc[i]),
                    need_cpu=float(sim.need_cpu[i]),
                    need_ram=float(sim.need_ram[i]),
                )
            )

        w = np.array(weights, dtype=float)
        total_w = float(w.sum())

        under_mask = np.array(
            [
                (cpu_alloc[i] < sim.need_cpu[i] - 1e-9)
                or (ram_alloc[i] < sim.need_ram[i] - 1e-9)
                for i in range(sim.n_vms)
            ],
            dtype=bool,
        )
        under_frac = float(w[under_mask].sum() / total_w) if total_w else 0.0

        over_samples = sum(v.cpu_over_hours + v.ram_over_hours for v in vm_risks)
        total_vm_hours = sim.n_vms * sim.n_samples
        over_ratio = over_samples / max(1, 2 * total_vm_hours)

        wasted_cpu = sum(v.wasted_cpu for v in vm_risks)
        wasted_ram = sum(v.wasted_ram for v in vm_risks)
        waste_frac = 0.5 * (
            wasted_cpu / max(1e-9, sim.host.total_cores)
            + wasted_ram / max(1e-9, sim.host.total_ram)
        )

        score = 100.0 * min(
            1.0, 0.40 * under_frac + 0.40 * min(1.0, over_ratio / 0.20) + 0.20 * waste_frac
        )

        unmet_cpu = sum(v.cpu_short for v in vm_risks)
        unmet_ram = sum(v.ram_short for v in vm_risks)
        mean_satisfaction = float(
            np.average([v.satisfaction for v in vm_risks], weights=w)
        )

        # ---- host level ---------------------------------------------------- #
        host = sim.host
        peak_c = float(sim.host_cpu.max())
        peak_r = float(sim.host_ram.max())
        host_reasons: List[str] = []

        alloc_c = float(cpu_alloc.sum())
        alloc_r = float(ram_alloc.sum())

        if alloc_c > host.total_cores + 1e-9 or alloc_r > host.total_ram + 1e-9:
            host_level = "critical"
            host_reasons.append(
                "The allocation exceeds the physical host capacity - VirtualBox "
                "would refuse it or the host would thrash."
            )
        else:
            head_c = host.total_cores - alloc_c
            head_r = host.total_ram - alloc_r
            if head_c < 0.5:
                host_reasons.append(
                    f"Only {head_c:.2f} vCPU of the {host.total_cores:g} host cores "
                    f"is left unassigned - no room for a new VM or a burst."
                )
            if head_r < 1024:
                host_reasons.append(
                    f"Only {head_r:.0f} MB of the {host.total_ram} MB host RAM is "
                    f"left unassigned - the host and VirtualBox itself need some."
                )
            if peak_c / host.total_cores >= 0.95:
                host_reasons.append(
                    f"Combined VM CPU demand peaks at {peak_c / host.total_cores:.0%} "
                    f"of the host - the machine is at saturation."
                )
            if peak_r / host.total_ram >= host.ram_alarm:
                host_reasons.append(
                    f"Combined VM RAM demand peaks at {peak_r / host.total_ram:.0%} "
                    f"of the host (alarm threshold {host.ram_alarm:.0%})."
                )
            worst = max(peak_c / host.total_cores, peak_r / host.total_ram)
            if worst >= 0.95 or head_r < 1024:
                host_level = "high"
            elif worst >= 0.85 or head_c < 1.0:
                host_level = "medium"
            else:
                host_level = "ok"

        host_risk = HostRisk(
            cpu_alloc=alloc_c,
            ram_alloc=alloc_r,
            cpu_total=float(host.total_cores),
            ram_total=float(host.total_ram),
            peak_demand_cpu=peak_c,
            peak_demand_ram=peak_r,
            cpu_alarm=host.cpu_alarm,
            ram_alarm=host.ram_alarm,
            level=host_level,
            reasons=host_reasons
            or ["Host has comfortable headroom for both CPU and RAM."],
        )

        counts = {lvl: 0 for lvl in LEVEL_ORDER}
        counts["waste"] = 0
        for v in vm_risks:
            counts[v.level] = counts.get(v.level, 0) + 1

        return RiskReport(
            label=label,
            vm_risks=vm_risks,
            host=host_risk,
            score=float(score),
            level=_level_from_score(score),
            counts=counts,
            overloaded_vm_hours=sum(v.over_hours for v in vm_risks),
            total_vm_hours=total_vm_hours,
            wasted_cpu=float(wasted_cpu),
            wasted_ram=float(wasted_ram),
            unmet_cpu=float(unmet_cpu),
            unmet_ram=float(unmet_ram),
            mean_satisfaction=mean_satisfaction,
        )


def detect_overload(
    sim: SimulationResult,
    cpu_alloc: Sequence[float],
    ram_alloc: Sequence[float],
    label: str = "allocation",
    detector: Optional[RiskDetector] = None,
) -> RiskReport:
    """Convenience wrapper around :class:`RiskDetector`."""
    detector = detector or RiskDetector()
    return detector.analyse(sim, cpu_alloc, ram_alloc, label=label)


# =========================================================================== #
# Self-test:  python monitor.py
# =========================================================================== #
if __name__ == "__main__":
    import sys

    try:                       # make emoji safe on legacy Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from optimizer import build_problem
    from simulator import simulate

    print("=" * 72)
    print("LIVE HOST (read-only, psutil)")
    print("=" * 72)
    mon = HostMonitor(interval=0.2)
    if mon.available:
        snap = mon.snapshot()
        print(f"CPU      : {snap.cpu_percent:5.1f} %   ({snap.logical_cpus} logical, "
              f"{snap.core_count} physical cores, load/core {snap.load_per_core:.2f})")
        print(f"RAM      : {snap.ram_percent:5.1f} %   ({snap.ram_used_mb:,.0f} / "
              f"{snap.ram_total_mb:,.0f} MB)")
        print(f"Severity : {LEVEL_EMOJI.get(snap.level, '')} {snap.level}")
        print("\nTop processes:")
        for name, cpu, mem in snap.top_processes:
            print(f"   {name[:28]:<30} CPU {cpu:6.1f} %   RAM {mem:8.1f} MB")
    else:
        print(mon.unavailable_reason())

    print()
    print("=" * 72)
    print("SIMULATED FLEET - before vs after")
    print("=" * 72)
    sim = simulate(days=7, seed=42)
    problem = build_problem(sim)
    detector = RiskDetector()

    before = detector.analyse(sim, problem.cur_cpu, problem.cur_ram, label="BEFORE")
    print(f"\nBEFORE  risk score {before.score:5.1f} ({before.level})   "
          f"overloaded VM-hours {before.overloaded_vm_hours}/{before.total_vm_hours}   "
          f"wasted {before.wasted_cpu:.2f} vCPU / {before.wasted_ram:.0f} MB")
    for v in before.vm_risks:
        print(f"   {LEVEL_EMOJI.get(v.level, '')} {v.name:<12} {v.level:<8} "
              f"cpu {v.cpu_util_peak * 100:5.0f}% ram {v.ram_util_peak * 100:5.0f}% "
              f"over {v.over_hours:3d} h")

    after = detector.analyse(
        sim, problem.cur_cpu * 0 + problem.req_cpu, problem.req_ram, label="AFTER"
    )
    print(f"\nAFTER (ideal: allocate exactly the measured peak)   "
          f"risk score {after.score:5.1f} ({after.level})   "
          f"overloaded VM-hours {after.overloaded_vm_hours}/{after.total_vm_hours}")
    for v in after.vm_risks:
        print(f"   {LEVEL_EMOJI.get(v.level, '')} {v.name:<12} {v.level:<8} "
              f"cpu {v.cpu_util_peak * 100:5.0f}% ram {v.ram_util_peak * 100:5.0f}% "
              f"over {v.over_hours:3d} h")

    from optimizer import HybridACOCSA

    hyb = HybridACOCSA(seed=42).optimize(problem)
    opt = detector.analyse(sim, hyb.best_cpu, hyb.best_ram, label="OPTIMISED")
    print(f"\nAFTER (Hybrid ACO-CSA)   risk score {opt.score:5.1f} ({opt.level})   "
          f"overloaded VM-hours {opt.overloaded_vm_hours}/{opt.total_vm_hours}   "
          f"wasted {opt.wasted_cpu:.2f} vCPU / {opt.wasted_ram:.0f} MB")
    for v in opt.vm_risks:
        print(f"   {LEVEL_EMOJI.get(v.level, '')} {v.name:<12} {v.level:<8} "
              f"cpu {v.cpu_util_peak * 100:5.0f}% ram {v.ram_util_peak * 100:5.0f}% "
              f"over {v.over_hours:3d} h")
