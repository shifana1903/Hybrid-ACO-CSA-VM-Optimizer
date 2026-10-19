"""
simulator.py -- Synthetic VirtualBox VM workload simulator.

SAFETY
------
This module is *purely synthetic*. It generates plausible CPU / RAM demand
curves for a small fleet of virtual machines. It does NOT import VirtualBox
bindings, does NOT start / stop / reconfigure any VM, and never touches the
host machine. It is the sandbox the rest of the project plays in.

The simulator models three workload shapes:

    "business"  office-hours traffic (peaks around 14:00, quiet at night,
                much quieter at weekends)
    "always"    24/7 services with a mild daytime swell (databases, agents)
    "batch"     nightly / early-morning batch jobs (build + test machines)

On top of the deterministic shape we add multiplicative log-normal noise and
occasional demand bursts, which is what creates realistic "overload" moments
for the risk detector to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PRIORITY_WEIGHT",
    "HostSpec",
    "VMProfile",
    "SimulationResult",
    "DEFAULT_HOST",
    "DEFAULT_PROFILES",
    "default_profiles",
    "random_profiles",
    "simulate",
    "duty_shape",
]


#: How much each VM class matters when the optimiser has to trade off.
PRIORITY_WEIGHT: Dict[str, float] = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


# --------------------------------------------------------------------------- #
# Hardware / machine descriptions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HostSpec:
    """The physical (or hypervisor) host that all VMs run on."""

    name: str = "vbox01 - VirtualBox Host"
    total_cores: float = 16.0          # total vCPU available to guest VMs
    total_ram: int = 20480             # MB of RAM available to guest VMs
    cpu_alarm: float = 0.85            # 85 % -> amber alert
    ram_alarm: float = 0.90            # 90 % -> amber alert

    @property
    def label(self) -> str:
        return (
            f"{self.name} - {self.total_cores:g} vCPU / "
            f"{self.total_ram / 1024:.0f} GB RAM"
        )


@dataclass(frozen=True)
class VMProfile:
    """Static description of one virtual machine."""

    name: str
    os: str = "Linux"
    current_cpu: float = 2.0           # vCPU currently assigned in VirtualBox
    current_ram: int = 2048            # MB currently assigned
    peak_cpu: float = 2.0              # worst-case CPU demand, in cores
    peak_ram: int = 2048               # worst-case RAM demand, in MB
    priority: str = "medium"           # critical | high | medium | low
    duty: str = "business"             # business | always | batch

    @property
    def weight(self) -> float:
        return PRIORITY_WEIGHT.get(self.priority, 2.0)

    @property
    def current_ram_gb(self) -> float:
        return self.current_ram / 1024.0


DEFAULT_HOST = HostSpec()

#: The hand-tuned demo scenario shipped with the dashboard.
#:   total current vCPU  = 2+4+3.5+3+2.5+1   = 16.0  (host is fully carved up)
#:   total peak   vCPU   = 2.7+5.0+4.1+1.0+0.7+1.5 = 15.0 (fits, but only if
#:                          the slices are handed out sensibly)
DEFAULT_PROFILES: Tuple[VMProfile, ...] = (
    VMProfile("web-server",  "Ubuntu 22.04", 2.0, 2048, 2.7, 2600, "high",     "business"),
    VMProfile("db-server",   "Ubuntu 22.04", 4.0, 6144, 5.0, 7000, "critical", "always"),
    VMProfile("app-server",  "Windows 11",   3.5, 4096, 4.1, 5000, "high",     "business"),
    VMProfile("build-agent", "Debian 12",    3.0, 3072, 1.0, 1800, "low",      "batch"),
    VMProfile("monitor-vm",  "Alpine 3.19",  2.5, 2048, 0.7, 1200, "medium",   "always"),
    VMProfile("test-vm",     "Windows 10",   1.0, 1024, 1.5, 1600, "medium",   "batch"),
)


def default_profiles() -> List[VMProfile]:
    """Return a fresh (mutable) copy of the hand-tuned demo scenario."""
    return list(DEFAULT_PROFILES)


# --------------------------------------------------------------------------- #
# Synthetic scenarios
# --------------------------------------------------------------------------- #
def random_profiles(n: int, host: HostSpec = DEFAULT_HOST, seed: int = 7) -> List[VMProfile]:
    """Build a random but *feasible* fleet of `n` VMs for `host`.

    Peak demands are drawn from Dirichlet distributions so that the whole
    fleet needs roughly 94 % of the host - tight enough to be interesting,
    loose enough that an optimal solution exists.
    """
    n = max(3, int(n))
    rng = np.random.default_rng(seed)

    wc = rng.dirichlet(np.ones(n))
    wr = rng.dirichlet(np.ones(n))

    peak_cpu = np.maximum(0.5, wc * host.total_cores * 0.94)
    peak_ram = np.maximum(512.0, wr * host.total_ram * 0.94)
    peak_cpu *= host.total_cores * 0.94 / peak_cpu.sum()
    peak_ram *= host.total_ram * 0.94 / peak_ram.sum()

    # Current (naive) allocations: a wobbly guess around the real demand.
    cur_cpu = np.maximum(0.5, np.round(peak_cpu * rng.uniform(0.55, 1.45, n) / 0.5) * 0.5)
    cur_ram = np.maximum(512.0, np.round(peak_ram * rng.uniform(0.55, 1.45, n) / 128.0) * 128.0)

    # Never hand out more than the host physically has.
    while cur_cpu.sum() > host.total_cores:
        i = int(np.argmax(cur_cpu))
        cur_cpu[i] = max(0.5, cur_cpu[i] - 0.5)
        if cur_cpu[i] <= 0.5 and cur_cpu.sum() > host.total_cores:
            break
    while cur_ram.sum() > host.total_ram:
        i = int(np.argmax(cur_ram))
        cur_ram[i] = max(512.0, cur_ram[i] - 128.0)
        if cur_ram[i] <= 512.0 and cur_ram.sum() > host.total_ram:
            break

    oses = ["Ubuntu 22.04", "Windows 11", "Debian 12", "Alpine 3.19", "Windows 10", "Rocky 9"]
    prios = ["critical", "high", "high", "medium", "medium", "low"]
    duties = ["always", "business", "business", "batch", "always", "batch"]

    return [
        VMProfile(
            name=f"vm-{i + 1:02d}",
            os=oses[i % len(oses)],
            current_cpu=float(cur_cpu[i]),
            current_ram=int(cur_ram[i]),
            peak_cpu=float(peak_cpu[i]),
            peak_ram=int(peak_ram[i]),
            priority=prios[i % len(prios)],
            duty=duties[i % len(duties)],
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Load shapes
# --------------------------------------------------------------------------- #
def duty_shape(hour_index: int, duty: str) -> float:
    """Normalised 0.05 .. 1.0 load multiplier for one hour of the series."""
    h = hour_index % 24
    day = hour_index // 24
    weekday = (day % 7) < 5

    if duty == "always":
        v = 0.60 + 0.35 * math.exp(-0.5 * ((h - 14) / 5.5) ** 2)
    elif duty == "batch":
        v = (
            0.30
            + 0.55 * math.exp(-0.5 * ((h - 4) / 3.0) ** 2)
            + 0.25 * math.exp(-0.5 * ((h - 20) / 3.0) ** 2)
        )
        if not weekday:
            v *= 0.80
    else:  # "business"
        v = 0.18 + 0.82 * math.exp(-0.5 * ((h - 14) / 4.2) ** 2)
        if not weekday:
            v *= 0.35

    return max(0.05, min(1.0, v))


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    """A simulated demand trace for a fleet of VMs on one host."""

    host: "HostSpec"
    profiles: List[VMProfile]
    hours: np.ndarray            # (T,)  sample index
    cpu: np.ndarray              # (N,T) CPU demand in cores
    ram: np.ndarray              # (N,T) RAM demand in MB
    seed: int
    days: int

    # ---- basic shape ----------------------------------------------------- #
    @property
    def n_vms(self) -> int:
        return len(self.profiles)

    @property
    def n_samples(self) -> int:
        return int(self.hours.size)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.profiles]

    @property
    def priorities(self) -> List[str]:
        return [p.priority for p in self.profiles]

    @property
    def current_cpu(self) -> np.ndarray:
        return np.array([p.current_cpu for p in self.profiles], dtype=float)

    @property
    def current_ram(self) -> np.ndarray:
        return np.array([p.current_ram for p in self.profiles], dtype=float)

    # ---- demand statistics ---------------------------------------------- #
    @property
    def need_cpu(self) -> np.ndarray:
        """Worst-case CPU demand per VM (cores)."""
        return self.cpu.max(axis=1)

    @property
    def need_ram(self) -> np.ndarray:
        """Worst-case RAM demand per VM (MB)."""
        return self.ram.max(axis=1)

    @property
    def mean_cpu(self) -> np.ndarray:
        return self.cpu.mean(axis=1)

    @property
    def mean_ram(self) -> np.ndarray:
        return self.ram.mean(axis=1)

    @property
    def host_cpu(self) -> np.ndarray:
        """Total CPU demand on the host at each sample (cores)."""
        return self.cpu.sum(axis=0)

    @property
    def host_ram(self) -> np.ndarray:
        """Total RAM demand on the host at each sample (MB)."""
        return self.ram.sum(axis=0)

    @property
    def peak_host_cpu_pct(self) -> float:
        return float(self.host_cpu.max() / self.host.total_cores)

    @property
    def peak_host_ram_pct(self) -> float:
        return float(self.host_ram.max() / self.host.total_ram)

    # ---- convenience ----------------------------------------------------- #
    def timeline_labels(self) -> List[str]:
        """Human friendly 'Day d HH:00' labels for the x axis."""
        return [
            f"D{h // 24 + 1} {h % 24:02d}:00"
            for h in self.hours.tolist()
        ]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def simulate(
    host: Optional[HostSpec] = None,
    profiles: Optional[Sequence[VMProfile]] = None,
    days: int = 7,
    seed: int = 42,
    bursts: bool = True,
) -> SimulationResult:
    """Generate a synthetic CPU / RAM demand trace.

    Parameters
    ----------
    host      : host capacity (defaults to the 16 vCPU / 20 GB demo host).
    profiles  : VM fleet (defaults to :data:`DEFAULT_PROFILES`).
    days      : length of the simulated window, in days (hourly samples).
    seed      : RNG seed - the same seed always reproduces the same trace.
    bursts    : inject occasional demand spikes (this is what causes the
                "overload / crash risk" the project is all about).
    """
    host = host or DEFAULT_HOST
    profiles = list(profiles) if profiles is not None else default_profiles()
    days = int(max(1, days))
    n = len(profiles)
    t = days * 24

    rng = np.random.default_rng(seed)
    hours = np.arange(t)

    cpu = np.zeros((n, t), dtype=float)
    ram = np.zeros((n, t), dtype=float)

    for i, p in enumerate(profiles):
        shape = np.array([duty_shape(h, p.duty) for h in hours])

        noise_cpu = rng.lognormal(0.0, 0.055, t)
        noise_ram = rng.lognormal(0.0, 0.035, t)

        # CPU: demand tracks the workload shape closely.
        d_cpu = p.peak_cpu * shape * noise_cpu

        # RAM: much stickier than CPU - it barely drops overnight.
        ram_shape = 0.74 + 0.26 * shape
        d_ram = p.peak_ram * ram_shape * noise_ram

        if bursts:
            hits = np.flatnonzero(rng.random(t) < 0.02)
            for h in hits:
                d_cpu[h] = max(d_cpu[h], p.peak_cpu * float(rng.uniform(0.95, 1.06)))
                d_ram[h] = max(d_ram[h], p.peak_ram * float(rng.uniform(0.95, 1.03)))

        cpu[i] = np.clip(d_cpu, 0.05 * p.peak_cpu, 1.06 * p.peak_cpu)
        ram[i] = np.clip(d_ram, 0.20 * p.peak_ram, 1.03 * p.peak_ram)

    return SimulationResult(
        host=host,
        profiles=profiles,
        hours=hours,
        cpu=cpu,
        ram=ram,
        seed=seed,
        days=days,
    )


# --------------------------------------------------------------------------- #
# Self-test when run directly:  python simulator.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sim = simulate(days=7, seed=42)
    print(f"Host        : {sim.host.label}")
    print(f"Samples     : {sim.n_samples} ({sim.days} days @ 1 h)")
    print(f"Peak host CPU demand : {sim.peak_host_cpu_pct:6.1%}")
    print(f"Peak host RAM demand : {sim.peak_host_ram_pct:6.1%}")
    print()
    hdr = f"{'VM':<13}{'pri':<9}{'nowCPU':>7}{'needCPU':>9}{'nowRAM':>8}{'needRAM':>9}"
    print(hdr)
    print("-" * len(hdr))
    for i, p in enumerate(sim.profiles):
        print(
            f"{p.name:<13}{p.priority:<9}{p.current_cpu:>7.1f}"
            f"{sim.need_cpu[i]:>9.2f}{p.current_ram:>8d}{sim.need_ram[i]:>9.0f}"
        )
    print(
        f"\nTOTAL current CPU {sim.current_cpu.sum():.1f} / {sim.host.total_cores:g} vCPU"
        f"   need {sim.need_cpu.sum():.2f}"
    )
    print(
        f"TOTAL current RAM {sim.current_ram.sum():.0f} / {sim.host.total_ram} MB"
        f"   need {sim.need_ram.sum():.0f}"
    )
