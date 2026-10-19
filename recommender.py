"""
recommender.py -- Turns optimiser output into human decisions.

Given the "before" risk report (what the VirtualBox fleet looks like today)
and the "after" risk report (what the Hybrid ACO-CSA solution would look
like), this module produces:

    * a one-line headline,
    * a prioritised list of alerts,
    * per-VM advice with the *reason* spelled out in plain numbers,
    * host-level notes (counted in vCPU and MB, not percentages),
    * a manual, step-by-step application plan for VirtualBox.

SAFETY
------
The plan is deliberately manual. This project never changes VirtualBox
settings, never starts/stops/restarts a VM and never writes to the host.
Every change is something a human reviews and applies by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from monitor import LEVEL_EMOJI, RiskReport, VMRisk, RiskDetector
from simulator import SimulationResult

__all__ = [
    "Alert",
    "Advice",
    "VMRow",
    "RecommendationReport",
    "RecommendationEngine",
    "SAFETY_NOTE",
]


SAFETY_NOTE = (
    "Simulation only. This tool never modifies VirtualBox configuration, never "
    "starts, stops or restarts a virtual machine, and never writes to the host. "
    "Every recommendation below is advisory - review it and apply it yourself."
)

_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}
_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Alert:
    severity: str          # CRITICAL | HIGH | MEDIUM | INFO
    target: str
    headline: str
    detail: str

    @property
    def emoji(self) -> str:
        return {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "INFO": "🔵",
        }.get(self.severity, "⚪")


@dataclass
class Advice:
    vm: str
    priority: str
    action: str            # INCREASE | REDUCE | REBALANCE | KEEP
    cpu_from: float
    cpu_to: float
    ram_from: float
    ram_to: float
    rationale: str

    @property
    def cpu_delta(self) -> float:
        return self.cpu_to - self.cpu_from

    @property
    def ram_delta(self) -> float:
        return self.ram_to - self.ram_from

    @property
    def changed(self) -> bool:
        return abs(self.cpu_delta) >= 0.05 or abs(self.ram_delta) >= 64

    def delta_text(self) -> str:
        bits = []
        if abs(self.cpu_delta) >= 0.05:
            bits.append(f"CPU {self.cpu_from:.2f} → {self.cpu_to:.2f} vCPU "
                        f"({self.cpu_delta:+.2f})")
        if abs(self.ram_delta) >= 64:
            bits.append(f"RAM {self.ram_from:.0f} → {self.ram_to:.0f} MB "
                        f"({self.ram_delta:+.0f})")
        return " · ".join(bits) if bits else "no change"


@dataclass
class VMRow:
    name: str
    priority: str
    level_before: str
    level_after: str
    need_cpu: float
    need_ram: float
    cpu_before: float
    cpu_after: float
    ram_before: float
    ram_after: float
    over_before: int
    over_after: int
    sat_before: float
    sat_after: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "VM": self.name,
            "Priority": self.priority,
            "Peak CPU (vCPU)": round(self.need_cpu, 2),
            "CPU now": round(self.cpu_before, 2),
            "CPU new": round(self.cpu_after, 2),
            "Peak RAM (MB)": round(self.need_ram),
            "RAM now": round(self.ram_before),
            "RAM new": round(self.ram_after),
            "Overloaded h (now)": self.over_before,
            "Overloaded h (new)": self.over_after,
            "Served now": f"{self.sat_before * 100:.0f} %",
            "Served new": f"{self.sat_after * 100:.0f} %",
            "Risk now": self.level_before,
            "Risk new": self.level_after,
        }


@dataclass
class RecommendationReport:
    method: str
    headline: str
    verdict: str
    score_before: float
    score_after: float
    level_before: str
    level_after: str
    alerts: List[Alert]
    advice: List[Advice]
    rows: List[VMRow]
    host_notes: List[str]
    plan: List[str]
    kpis: Dict[str, float] = field(default_factory=dict)
    safety: str = SAFETY_NOTE

    # -- convenience -------------------------------------------------------- #
    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before

    @property
    def improvements(self) -> List[Advice]:
        return [a for a in self.advice if a.action == "INCREASE"]

    @property
    def reductions(self) -> List[Advice]:
        return [a for a in self.advice if a.action == "REDUCE"]

    def table(self) -> List[Dict[str, object]]:
        return [r.to_dict() for r in self.rows]

    def summary(self) -> str:
        """A compact plain-text report (used by the CLI and by ``app.py``)."""
        lines: List[str] = []
        lines.append(self.headline)
        lines.append(self.verdict)
        lines.append("")

        if self.alerts:
            lines.append("RISK ALERTS (before optimisation)")
            for a in sorted(self.alerts, key=lambda a: _SEVERITY_ORDER.get(a.severity, 9)):
                lines.append(f"  {a.emoji} [{a.severity}] {a.target}: {a.headline}")
                lines.append(f"      {a.detail}")
            lines.append("")

        lines.append("RECOMMENDED CHANGES")
        for a in sorted(self.advice, key=lambda a: _PRIORITY_ORDER.get(a.priority, 9)):
            lines.append(f"  [{a.priority.upper():<8}] {a.vm:<13} {a.action:<9} {a.delta_text()}")
            lines.append(f"      {a.rationale}")
        lines.append("")

        if self.host_notes:
            lines.append("HOST NOTES")
            for note in self.host_notes:
                lines.append(f"  - {note}")
            lines.append("")

        lines.append("HOW TO APPLY (manual, one VM at a time)")
        for i, step in enumerate(self.plan, start=1):
            lines.append(f"  {i}. {step}")
        lines.append("")
        lines.append(f"SAFETY: {self.safety}")
        return "\n".join(lines)


class RecommendationEngine:
    """Builds an explainable report from a before/after risk comparison."""

    def __init__(
        self,
        sim: SimulationResult,
        detector: Optional[RiskDetector] = None,
    ) -> None:
        self.sim = sim
        self.detector = detector or RiskDetector()

    # ------------------------------------------------------------------ #
    def build(
        self,
        before: RiskReport,
        after: RiskReport,
        method: str = "Hybrid ACO-CSA",
        iterations: Optional[int] = None,
        evaluations: Optional[int] = None,
    ) -> RecommendationReport:
        rows = self._rows(before, after)
        advice = [self._advice(r, before.by_name()[r.name], after.by_name()[r.name])
                  for r in rows]
        alerts = self._alerts(before)
        host_notes = self._host_notes(before, after)

        delta = before.score - after.score
        if delta >= 40:
            verdict = ("Major improvement: the fleet moves from dangerous "
                       "over-commitment to a sustainable allocation.")
        elif delta >= 15:
            verdict = "Solid improvement: overload risk is clearly reduced."
        elif delta > 3:
            verdict = "Modest improvement: risk edges down, review the remaining alerts."
        elif delta >= -3:
            verdict = ("Little change - the current allocation is already close to "
                       "what the demand supports, or the host is simply too small.")
        else:
            verdict = ("The optimiser could not beat the current allocation with these "
                       "settings. Try more iterations, a different seed, or check "
                       "whether the fleet really fits on this host.")

        headline = (
            f"Overload-risk score {before.score:.0f} → {after.score:.0f} / 100 "
            f"({LEVEL_EMOJI.get(before.level, '')} {before.level} → "
            f"{LEVEL_EMOJI.get(after.level, '')} {after.level}); "
            f"overloaded VM-hours {before.overloaded_vm_hours} → "
            f"{after.overloaded_vm_hours} out of {before.total_vm_hours}."
        )

        kpis = {
            "risk_before": before.score,
            "risk_after": after.score,
            "risk_delta": delta,
            "overload_hours_before": before.overloaded_vm_hours,
            "overload_hours_after": after.overloaded_vm_hours,
            "wasted_cpu_before": before.wasted_cpu,
            "wasted_cpu_after": after.wasted_cpu,
            "wasted_ram_before": before.wasted_ram,
            "wasted_ram_after": after.wasted_ram,
            "satisfaction_before": before.mean_satisfaction,
            "satisfaction_after": after.mean_satisfaction,
            "unmet_cpu_before": before.unmet_cpu,
            "unmet_cpu_after": after.unmet_cpu,
            "unmet_ram_before": before.unmet_ram,
            "unmet_ram_after": after.unmet_ram,
            "cpu_reallocated": float(
                np.abs(
                    np.array([a.cpu_delta for a in advice], dtype=float)
                ).sum()
                / 2.0
            ),
            "ram_reallocated": float(
                np.abs(
                    np.array([a.ram_delta for a in advice], dtype=float)
                ).sum()
                / 2.0
            ),
        }
        if iterations is not None:
            kpis["iterations"] = float(iterations)
        if evaluations is not None:
            kpis["evaluations"] = float(evaluations)

        return RecommendationReport(
            method=method,
            headline=headline,
            verdict=verdict,
            score_before=before.score,
            score_after=after.score,
            level_before=before.level,
            level_after=after.level,
            alerts=alerts,
            advice=advice,
            rows=rows,
            host_notes=host_notes,
            plan=self._plan(before, after, advice),
            kpis=kpis,
        )

    # ------------------------------------------------------------------ #
    def _rows(self, before: RiskReport, after: RiskReport) -> List[VMRow]:
        b = before.by_name()
        a = after.by_name()
        rows: List[VMRow] = []
        for i, profile in enumerate(self.sim.profiles):
            vb, va = b[profile.name], a[profile.name]
            rows.append(
                VMRow(
                    name=profile.name,
                    priority=profile.priority,
                    level_before=vb.level,
                    level_after=va.level,
                    need_cpu=float(self.sim.need_cpu[i]),
                    need_ram=float(self.sim.need_ram[i]),
                    cpu_before=vb.alloc_cpu,
                    cpu_after=va.alloc_cpu,
                    ram_before=vb.alloc_ram,
                    ram_after=va.alloc_ram,
                    over_before=vb.over_hours,
                    over_after=va.over_hours,
                    sat_before=vb.satisfaction,
                    sat_after=va.satisfaction,
                )
            )
        return rows

    # ------------------------------------------------------------------ #
    def _advice(self, row: VMRow, rb: VMRisk, ra: VMRisk) -> Advice:
        dc = row.cpu_after - row.cpu_before
        dr = row.ram_after - row.ram_before

        cpu_up, cpu_dn = dc >= 0.05, dc <= -0.05
        ram_up, ram_dn = dr >= 64, dr <= -64

        if not (cpu_up or cpu_dn or ram_up or ram_dn):
            action = "KEEP"
        elif (cpu_up or ram_up) and (cpu_dn or ram_dn):
            action = "REBALANCE"
        elif cpu_up or ram_up:
            action = "INCREASE"
        else:
            action = "REDUCE"

        horizon = self.sim.n_samples
        reasons: List[str] = []

        if rb.over_hours > 0:
            reasons.append(
                f"currently starved for {rb.over_hours} of {horizon} hours "
                f"(CPU peaked at {rb.cpu_util_peak * 100:.0f} % of the assigned "
                f"{rb.alloc_cpu:.2f} vCPU, RAM at {rb.ram_util_peak * 100:.0f} % "
                f"of {rb.alloc_ram:.0f} MB)"
            )
        if rb.wasted_cpu > 0.75 or rb.wasted_ram > 512:
            reasons.append(
                f"currently over-provisioned by {rb.wasted_cpu:.2f} vCPU / "
                f"{rb.wasted_ram:.0f} MB that is never touched"
            )
        if not reasons:
            reasons.append(
                f"already healthy (peak CPU {rb.cpu_util_peak * 100:.0f} %, "
                f"peak RAM {rb.ram_util_peak * 100:.0f} %)"
            )

        need = (
            f"measured peak demand is {row.need_cpu:.2f} vCPU and "
            f"{row.need_ram:.0f} MB"
        )

        tail = ""
        if action != "KEEP":
            tail = (
                f" New allocation serves "
                f"{ra.satisfaction * 100:.0f} % of demand (was "
                f"{rb.satisfaction * 100:.0f} %) with "
                f"{ra.over_hours} overloaded hours left."
            )
        else:
            tail = " No change recommended."

        rationale = (
            f"[{row.priority} priority] {need}; {reasons[0]}." + tail
        )

        return Advice(
            vm=row.name,
            priority=row.priority,
            action=action,
            cpu_from=row.cpu_before,
            cpu_to=row.cpu_after,
            ram_from=row.ram_before,
            ram_to=row.ram_after,
            rationale=rationale,
        )

    # ------------------------------------------------------------------ #
    def _alerts(self, before: RiskReport) -> List[Alert]:
        alerts: List[Alert] = []
        horizon = self.sim.n_samples

        for v in before.vm_risks:
            if v.level == "critical":
                alerts.append(
                    Alert(
                        severity="CRITICAL",
                        target=v.name,
                        headline=(
                            f"{v.name} is starved: CPU peak "
                            f"{v.cpu_util_peak * 100:.0f} % / RAM peak "
                            f"{v.ram_util_peak * 100:.0f} % of its slice"
                        ),
                        detail=(
                            f"{v.over_hours} of {horizon} hours exceed the assigned "
                            f"resources ({v.alloc_cpu:.2f} vCPU / {v.alloc_ram:.0f} MB) "
                            f"while the guest actually wants {v.need_cpu:.2f} vCPU / "
                            f"{v.need_ram:.0f} MB. Priority: {v.priority}. This is the "
                            f"classic path to a hung or crashed VM."
                        ),
                    )
                )
            elif v.level == "high":
                alerts.append(
                    Alert(
                        severity="HIGH",
                        target=v.name,
                        headline=f"{v.name} has overloaded hours in the trace",
                        detail=(
                            f"{v.over_hours} hour(s) exceed the assigned resources. "
                            f"Small, but it is the early-warning signal."
                        ),
                    )
                )
            elif v.level == "waste":
                alerts.append(
                    Alert(
                        severity="INFO",
                        target=v.name,
                        headline=(
                            f"{v.name} wastes {v.wasted_cpu:.2f} vCPU / "
                            f"{v.wasted_ram:.0f} MB"
                        ),
                        detail=(
                            f"Assigned {v.alloc_cpu:.2f} vCPU / {v.alloc_ram:.0f} MB but "
                            f"never uses more than {v.need_cpu:.2f} vCPU / "
                            f"{v.need_ram:.0f} MB. That stranded capacity can fund "
                            f"another VM."
                        ),
                    )
                )

        host = before.host
        if host.level in ("critical", "high"):
            alerts.append(
                Alert(
                    severity="CRITICAL" if host.level == "critical" else "HIGH",
                    target="HOST",
                    headline=(
                        f"Host is tight: {host.cpu_alloc:.1f}/{host.cpu_total:g} vCPU and "
                        f"{host.ram_alloc:.0f}/{host.ram_total:.0f} MB assigned"
                    ),
                    detail=" ".join(host.reasons),
                )
            )
        return alerts

    # ------------------------------------------------------------------ #
    def _host_notes(self, before: RiskReport, after: RiskReport) -> List[str]:
        hb, ha = before.host, after.host
        host = self.sim.host
        notes: List[str] = []

        notes.append(
            f"vCPU assigned: {hb.cpu_alloc:.2f} → {ha.cpu_alloc:.2f} of "
            f"{host.total_cores:g} ({ha.cpu_alloc_pct:.0%} of the host, "
            f"{ha.cpu_headroom:+.2f} vCPU spare)."
        )
        notes.append(
            f"RAM assigned: {hb.ram_alloc:.0f} → {ha.ram_alloc:.0f} MB of "
            f"{host.total_ram} ({ha.ram_alloc_pct:.0%} of the host, "
            f"{ha.ram_headroom:+.0f} MB spare)."
        )
        notes.append(
            f"Combined peak demand of all guests is "
            f"{ha.peak_demand_cpu:.2f} vCPU ({ha.peak_demand_cpu_pct:.0%}) and "
            f"{ha.peak_demand_ram:.0f} MB ({ha.peak_demand_ram_pct:.0%}) of the host. "
            f"That demand does not change with allocation - what changes is whether "
            f"each guest gets its share."
        )
        notes.append(
            f"Stranded capacity falls from {before.wasted_cpu:.2f} vCPU / "
            f"{before.wasted_ram:.0f} MB to {after.wasted_cpu:.2f} vCPU / "
            f"{after.wasted_ram:.0f} MB."
        )
        for reason in ha.reasons:
            if reason.startswith("Host has comfortable"):
                notes.append(reason)
            else:
                notes.append(f"⚠️ {reason}")
        return notes

    # ------------------------------------------------------------------ #
    def _plan(
        self,
        before: RiskReport,
        after: RiskReport,
        advice: Sequence[Advice],
    ) -> List[str]:
        increases = [a for a in advice if a.cpu_delta > 0.05 or a.ram_delta > 64]
        reductions = [a for a in advice if a.cpu_delta < -0.05 or a.ram_delta < -64]
        reductions.sort(key=lambda a: _PRIORITY_ORDER.get(a.priority, 9), reverse=True)
        increases.sort(key=lambda a: _PRIORITY_ORDER.get(a.priority, 9))

        plan: List[str] = [
            "Take a VirtualBox snapshot of every VM you are about to touch, so the "
            "change can be rolled back.",
            "Record the 'before' numbers from this report (risk score "
            f"{before.score:.0f}/100, {before.overloaded_vm_hours} overloaded VM-hours).",
        ]

        if reductions:
            first = reductions[0]
            plan.append(
                "Free up capacity FIRST by reducing the over-provisioned guests: "
                + "; ".join(
                    f"{a.vm} → {a.cpu_to:.2f} vCPU / {a.ram_to:.0f} MB" for a in reductions
                )
                + f". Start with {first.vm} (lowest priority first)."
            )
        if increases:
            plan.append(
                "Then grow the starved guests: "
                + "; ".join(
                    f"{a.vm} → {a.cpu_to:.2f} vCPU / {a.ram_to:.0f} MB"
                    for a in increases[:4]
                )
                + ("." if len(increases) <= 4 else f" (and {len(increases) - 4} more).")
            )

        plan.extend(
            [
                "In VirtualBox Manager: select the VM → Settings → System → "
                "Motherboard → Base Memory (MB), then System → Processor → "
                "Processor(s) (vCPU). Shut the guest down gracefully first - do NOT "
                "use 'Reset' or pull the power.",
                f"Keep the sum of all guests at or below the host budget: "
                f"{self.sim.host.total_cores:g} vCPU and "
                f"{self.sim.host.total_ram} MB, leaving some headroom for the host "
                "and for VirtualBox itself.",
                "Re-run the simulator + optimiser (or the dashboard) after the change "
                "and confirm the risk score came down and no VM is still overloaded.",
                "Apply changes one VM at a time and watch the guest for a day before "
                "touching the next one.",
                "Nothing in this project applies the change for you - all edits are "
                "manual and reversible.",
            ]
        )
        return plan


# --------------------------------------------------------------------------- #
# Self-test:  python recommender.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from monitor import RiskDetector
    from optimizer import HybridACOCSA, build_problem
    from simulator import simulate

    sim = simulate(days=7, seed=42)
    problem = build_problem(sim)
    detector = RiskDetector()

    before = detector.analyse(sim, problem.cur_cpu, problem.cur_ram, label="BEFORE")
    result = HybridACOCSA(seed=42).optimize(problem)
    after = detector.analyse(sim, result.best_cpu, result.best_ram, label="AFTER")

    engine = RecommendationEngine(sim, detector)
    report = engine.build(
        before,
        after,
        method=result.method,
        iterations=len(result.history),
        evaluations=result.evaluations,
    )
    print(report.summary())
    print()
    print("KPI:", {k: round(v, 3) for k, v in report.kpis.items()})
