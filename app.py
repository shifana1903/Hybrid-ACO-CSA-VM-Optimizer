"""
app.py -- Streamlit dashboard for the Hybrid ACO-CSA VirtualBox optimiser.

The dashboard is read-only and simulation-first:

    * it simulates a VirtualBox VM fleet (CPU / RAM demand over time),
    * scores the *current* allocation for overload risk,
    * runs ACO, Cuckoo Search and the Hybrid ACO-CSA optimiser,
    * scores the *optimised* allocation,
    * and explains every recommended change in plain numbers.

Run it with:      streamlit run app.py

SAFETY: this dashboard never modifies VirtualBox, never starts/stops/restarts
a VM and never writes to the host. psutil is used in read-only mode.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from monitor import LEVEL_EMOJI, HostMonitor, RiskDetector, RiskReport
from optimizer import (
    ACOOptimizer,
    CSAOptimizer,
    HybridACOCSA,
    OptimizationResult,
    build_problem,
)
from recommender import RecommendationEngine, SAFETY_NOTE
from simulator import DEFAULT_HOST, default_profiles, random_profiles, simulate

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Hybrid ACO-CSA · VirtualBox VM Resource Optimiser",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706",
    "#7c3aed", "#0891b2", "#db2777", "#65a30d",
]
BAD, GOOD, WARN, NEUTRAL = "#dc2626", "#16a34a", "#d97706", "#94a3b8"


def _streamlit_at_least(major: int, minor: int) -> bool:
    parts = getattr(st, "__version__", "0.0").split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):                      # pragma: no cover
        return False


#: Streamlit renamed ``use_container_width`` to ``width="stretch"`` in 1.49.
_WIDE = {"width": "stretch"} if _streamlit_at_least(1, 49) else {"use_container_width": True}

ALGORITHM_KEYS = {
    "Hybrid ACO-CSA (recommended)": ["Hybrid ACO-CSA"],
    "ACO only": ["ACO"],
    "CSA only": ["CSA"],
    "Compare all three": ["ACO", "CSA", "Hybrid ACO-CSA"],
}


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_sim(days: int, seed: int, n_vms: int, scenario_seed: int):
    """Simulate the fleet (cached - only re-runs when a knob changes)."""
    if n_vms == 6:
        profiles = default_profiles()
    else:
        profiles = random_profiles(n_vms, DEFAULT_HOST, seed=scenario_seed)
    return simulate(profiles=profiles, days=days, seed=seed)


@st.cache_data(show_spinner=False)
def run_optimizers(
    days: int,
    seed: int,
    n_vms: int,
    scenario_seed: int,
    aco_ants: int,
    aco_iterations: int,
    csa_nests: int,
    csa_generations: int,
    algorithms: List[str],
) -> Dict[str, OptimizationResult]:
    """Run the requested metaheuristics (cached)."""
    sim = load_sim(days, seed, n_vms, scenario_seed)
    problem = build_problem(sim)
    out: Dict[str, OptimizationResult] = {}

    if "ACO" in algorithms:
        out["ACO"] = ACOOptimizer(
            n_ants=aco_ants, iterations=aco_iterations, seed=seed
        ).optimize(problem, np.random.default_rng(seed))

    if "CSA" in algorithms:
        out["CSA"] = CSAOptimizer(
            n_nests=csa_nests, generations=csa_generations, seed=seed
        ).optimize(problem, rng=np.random.default_rng(seed))

    if "Hybrid ACO-CSA" in algorithms:
        out["Hybrid ACO-CSA"] = HybridACOCSA(
            aco_ants=aco_ants,
            aco_iterations=aco_iterations,
            csa_nests=csa_nests,
            csa_generations=csa_generations,
            seed=seed,
        ).optimize(problem, np.random.default_rng(seed))

    return out


# --------------------------------------------------------------------------- #
# Chart helpers (Matplotlib)
# --------------------------------------------------------------------------- #
def _finish(fig, axes) -> None:
    for ax in np.atleast_1d(axes).ravel():
        ax.grid(alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def fig_host_load(sim) -> plt.Figure:
    host = sim.host
    h = sim.hours
    cpu_pct = sim.host_cpu / host.total_cores * 100.0
    ram_pct = sim.host_ram / host.total_ram * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(h, cpu_pct, lw=1.3, color=PALETTE[0], label="total VM CPU demand")
    axes[0].axhline(100, ls=":", lw=1.2, color=BAD, label="host capacity")
    axes[0].axhline(host.cpu_alarm * 100, ls="--", lw=1.2, color=WARN,
                    label=f"alarm {host.cpu_alarm:.0%}")
    axes[0].fill_between(h, host.cpu_alarm * 100, cpu_pct,
                         where=cpu_pct > host.cpu_alarm * 100,
                         color=BAD, alpha=0.15, interpolate=True)
    axes[0].set_ylabel("CPU (% of host)")

    axes[1].plot(h, ram_pct, lw=1.3, color=PALETTE[2], label="total VM RAM demand")
    axes[1].axhline(100, ls=":", lw=1.2, color=BAD, label="host capacity")
    axes[1].axhline(host.ram_alarm * 100, ls="--", lw=1.2, color=WARN,
                    label=f"alarm {host.ram_alarm:.0%}")
    axes[1].fill_between(h, host.ram_alarm * 100, ram_pct,
                         where=ram_pct > host.ram_alarm * 100,
                         color=BAD, alpha=0.15, interpolate=True)
    axes[1].set_ylabel("RAM (% of host)")
    axes[1].set_xlabel("simulated hour")

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Host aggregate demand · {host.label}")
    _finish(fig, axes)
    fig.tight_layout()
    return fig


def fig_vm_utilisation(sim, cpu_alloc, ram_alloc, title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for i, profile in enumerate(sim.profiles):
        colour = PALETTE[i % len(PALETTE)]
        axes[0].plot(sim.hours, sim.cpu[i] / max(cpu_alloc[i], 1e-9) * 100.0,
                     lw=1.1, color=colour, label=profile.name)
        axes[1].plot(sim.hours, sim.ram[i] / max(ram_alloc[i], 1e-9) * 100.0,
                     lw=1.1, color=colour, label=profile.name)

    for ax, label in zip(axes, ("CPU", "RAM")):
        ax.axhline(100, color=BAD, ls="--", lw=1.4)
        ax.set_ylabel(f"{label} used (% of slice)")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[0].set_title(title)
    axes[1].set_xlabel("simulated hour")
    _finish(fig, axes)
    fig.tight_layout()
    return fig


def fig_allocation(sim, before_cpu, before_ram, after_cpu, after_ram) -> plt.Figure:
    names = sim.names
    need_cpu, need_ram = sim.need_cpu, sim.need_ram
    x = np.arange(len(names))
    w = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(x - w, before_cpu, w, label="assigned now", color=NEUTRAL)
    axes[0].bar(x, need_cpu, w, label="measured peak demand", color=BAD, alpha=0.85)
    axes[0].bar(x + w, after_cpu, w, label="optimised", color=GOOD)
    axes[0].set_title("CPU allocation (vCPU)")
    axes[0].set_ylabel("vCPU")

    axes[1].bar(x - w, before_ram, w, label="assigned now", color=NEUTRAL)
    axes[1].bar(x, need_ram, w, label="measured peak demand", color=BAD, alpha=0.85)
    axes[1].bar(x + w, after_ram, w, label="optimised", color=GOOD)
    axes[1].set_title("RAM allocation (MB)")
    axes[1].set_ylabel("MB")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=8)
    _finish(fig, axes)
    fig.tight_layout()
    return fig


def fig_convergence(results: Dict[str, OptimizationResult]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (name, res) in enumerate(results.items()):
        ax.plot(range(1, len(res.history) + 1), res.history, lw=1.9,
                color=PALETTE[i % len(PALETTE)],
                label=f"{name} — best {res.best_fitness:.4g}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel("best fitness (lower is better)")
    ax.set_title("Convergence — the hybrid starts from ACO's elite solutions")
    ax.legend(fontsize=9)
    _finish(fig, ax)
    fig.tight_layout()
    return fig


def fig_risk(before: RiskReport, after: RiskReport) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    bars = axes[0].bar(["before", "after"], [before.score, after.score],
                       color=[BAD, GOOD], width=0.5)
    for bar, value in zip(bars, (before.score, after.score)):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                     f"{value:.0f}", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("overload-risk score (0–100)")
    axes[0].set_title("Risk score")

    labels = ["critical", "high", "waste", "ok"]
    colours = [BAD, WARN, NEUTRAL, GOOD]
    bottom_b = bottom_a = 0.0
    for label, colour in zip(labels, colours):
        cb = before.counts.get(label, 0)
        ca = after.counts.get(label, 0)
        axes[1].bar(["before"], [cb], bottom=[bottom_b], color=colour, label=label)
        axes[1].bar(["after"], [ca], bottom=[bottom_a], color=colour)
        bottom_b += cb
        bottom_a += ca
    axes[1].set_ylabel("number of VMs")
    axes[1].set_title("VM risk classes")
    axes[1].legend(fontsize=8, title="risk class")

    _finish(fig, axes)
    fig.tight_layout()
    return fig


def fig_satisfaction(before: RiskReport, after: RiskReport, names) -> plt.Figure:
    b = [v.satisfaction * 100 for v in before.vm_risks]
    a = [v.satisfaction * 100 for v in after.vm_risks]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - w / 2, b, w, label="before", color=NEUTRAL)
    ax.bar(x + w / 2, a, w, label="after", color=GOOD)
    ax.axhline(100, ls="--", lw=1.1, color=BAD)
    ax.set_ylim(0, 112)
    ax.set_ylabel("demand served (%)")
    ax.set_title("How much of each VM's demand the allocation actually serves")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    _finish(fig, ax)
    fig.tight_layout()
    return fig


def fig_waste(before: RiskReport, after: RiskReport) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(["before", "after"], [before.wasted_cpu, after.wasted_cpu],
                color=[BAD, GOOD], width=0.5)
    axes[0].set_ylabel("vCPU")
    axes[0].set_title("Stranded (assigned but never used) vCPU")
    axes[1].bar(["before", "after"], [before.wasted_ram, after.wasted_ram],
                color=[BAD, GOOD], width=0.5)
    axes[1].set_ylabel("MB")
    axes[1].set_title("Stranded RAM")
    _finish(fig, axes)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧮 Hybrid ACO-CSA")
    st.caption("VirtualBox VM resource optimiser — **simulation mode only**")

    st.subheader("1 · Fleet")
    scenario = st.selectbox(
        "Scenario",
        ["Demo fleet (6 named VMs)", "Random fleet (4 VMs)", "Random fleet (8 VMs)"],
        index=0,
    )
    n_vms = 6 if scenario.startswith("Demo") else (4 if "4" in scenario else 8)

    days = st.slider("Simulation window (days, 1 h samples)", 3, 14, 7)
    trace_seed = st.number_input("Trace seed", 0, 9999, 42, step=1)
    scenario_seed = st.number_input("Fleet seed (random scenarios only)",
                                    0, 9999, 7, step=1)

    st.subheader("2 · Optimiser")
    choice = st.radio("Algorithm", list(ALGORITHM_KEYS), index=0)
    aco_ants = st.slider("ACO ants", 5, 60, 20)
    aco_iterations = st.slider("ACO iterations", 5, 120, 40)
    csa_nests = st.slider("CSA nests", 5, 60, 24)
    csa_generations = st.slider("CSA generations", 20, 400, 160)
    opt_seed = st.number_input("Optimiser seed", 0, 9999, 42, step=1)

    st.divider()
    if st.button("🔄 Clear cache & re-run", **_WIDE):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.error(
        "**Safety**\n\n"
        "Simulation only. This app never changes VirtualBox settings, never "
        "starts/stops/restarts a VM and never writes to the host.",
        icon="🛡️",
    )


# --------------------------------------------------------------------------- #
# Compute everything
# --------------------------------------------------------------------------- #
sim = load_sim(int(days), int(trace_seed), int(n_vms), int(scenario_seed))
problem = build_problem(sim)
detector = RiskDetector()

before = detector.analyse(sim, problem.cur_cpu, problem.cur_ram, label="BEFORE")

algorithms = ALGORITHM_KEYS[choice]
results = run_optimizers(
    int(days), int(trace_seed), int(n_vms), int(scenario_seed),
    int(aco_ants), int(aco_iterations), int(csa_nests), int(csa_generations),
    algorithms,
)
primary_name = "Hybrid ACO-CSA" if "Hybrid ACO-CSA" in results else algorithms[0]
primary = results[primary_name]
after = detector.analyse(sim, primary.best_cpu, primary.best_ram, label="AFTER")

engine = RecommendationEngine(sim, detector)
report = engine.build(
    before,
    after,
    method=primary.method,
    iterations=len(primary.history),
    evaluations=primary.evaluations,
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Hybrid ACO-CSA · VirtualBox VM Resource Optimiser")
st.caption(
    f"{sim.host.label} · {sim.n_vms} VMs · {sim.n_samples} hourly samples "
    f"({sim.days} days) · optimiser: **{primary.method}**"
)
st.info(
    "**Simulation mode.** All numbers come from a synthetic workload model. "
    "No VirtualBox setting is read or written, and no VM is started, stopped or "
    "restarted. Recommendations are advisory — apply them by hand.",
    icon="🛡️",
)

k = report.kpis
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Risk score (before)", f"{before.score:.0f} / 100",
          help="0 = no VM overloaded, 100 = worst case.")
c2.metric("Risk score (after)", f"{after.score:.0f} / 100",
          delta=f"{after.score - before.score:+.0f}",
          delta_color="inverse")
c3.metric("Overloaded VM-hours", f"{after.overloaded_vm_hours}",
          delta=f"{after.overloaded_vm_hours - before.overloaded_vm_hours}",
          delta_color="inverse",
          help=f"out of {before.total_vm_hours} VM-hours in the trace")
c4.metric("Stranded vCPU", f"{after.wasted_cpu:.2f}",
          delta=f"{after.wasted_cpu - before.wasted_cpu:+.2f}",
          delta_color="inverse")
c5.metric("Demand served", f"{after.mean_satisfaction * 100:.0f} %",
          delta=f"{(after.mean_satisfaction - before.mean_satisfaction) * 100:+.0f} pts")

st.success(report.headline)
st.write(f"**{report.verdict}**")

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs([
    "📊 Overview",
    "⚠️ Risk alerts",
    "📈 Workload",
    "🧠 Optimisation",
    "🔄 Before vs After",
    "💡 Recommendations",
    "🖥️ Host monitor (psutil)",
])

# --- 1. Overview ----------------------------------------------------------- #
with tabs[0]:
    st.subheader("Host load over the simulated window")
    st.pyplot(fig_host_load(sim))
    plt.close("all")
    st.caption(
        "The host is already tight: combined demand peaks at "
        f"{sim.peak_host_cpu_pct:.0%} of the CPU and "
        f"{sim.peak_host_ram_pct:.0%} of the RAM. Note that this curve does not "
        "change when you re-allocate — what changes is whether each guest gets "
        "the slice it needs."
    )
    st.divider()
    st.subheader("Risk: before vs after")
    st.pyplot(fig_risk(before, after))
    plt.close("all")

# --- 2. Risk alerts -------------------------------------------------------- #
with tabs[1]:
    st.subheader("Alerts raised on the current allocation")
    if report.alerts:
        for alert in report.alerts:
            text = f"{alert.emoji} **[{alert.severity}] {alert.target}** — {alert.headline}\n\n{alert.detail}"
            if alert.severity == "CRITICAL":
                st.error(text)
            elif alert.severity == "HIGH":
                st.warning(text)
            else:
                st.info(text)
    else:
        st.success("No alerts — every VM currently gets the resources it needs.")

    st.divider()
    st.subheader("Per-VM risk table")
    rows = []
    for v in before.vm_risks:
        rows.append({
            "VM": f"{LEVEL_EMOJI.get(v.level, '')} {v.name}",
            "Priority": v.priority,
            "Risk": v.level,
            "Assigned vCPU": f"{v.alloc_cpu:.2f}",
            "Peak demand vCPU": f"{v.need_cpu:.2f}",
            "Assigned MB": f"{v.alloc_ram:.0f}",
            "Peak demand MB": f"{v.need_ram:.0f}",
            "CPU peak %": f"{v.cpu_util_peak * 100:.0f}",
            "RAM peak %": f"{v.ram_util_peak * 100:.0f}",
            "Overloaded hours": v.over_hours,
            "Wasted vCPU": f"{v.wasted_cpu:.2f}",
            "Wasted MB": f"{v.wasted_ram:.0f}",
        })
    st.dataframe(rows, **_WIDE, hide_index=True)

    with st.expander("Why each VM got that rating"):
        for v in before.vm_risks:
            st.markdown(f"**{LEVEL_EMOJI.get(v.level, '')} {v.name}** "
                        f"({v.priority} priority) — `{v.level}`")
            for reason in v.reasons:
                st.markdown(f"- {reason}")

# --- 3. Workload ----------------------------------------------------------- #
with tabs[2]:
    st.subheader("Demand vs the current allocation")
    st.pyplot(fig_vm_utilisation(
        sim, problem.cur_cpu, problem.cur_ram,
        "Per-VM utilisation of the CURRENT allocation "
        "(anything above the red line = the guest is being starved)",
    ))
    plt.close("all")

    st.subheader("Demand vs the optimised allocation")
    st.pyplot(fig_vm_utilisation(
        sim, primary.best_cpu, primary.best_ram,
        f"Per-VM utilisation under the {primary.method} allocation",
    ))
    plt.close("all")

    with st.expander("Raw simulated demand (first 48 hours)"):
        show = min(48, sim.n_samples)
        st.dataframe(
            {
                "hour": [f"D{h // 24 + 1} {h % 24:02d}:00" for h in sim.hours[:show]],
                **{
                    f"{p.name} CPU": np.round(sim.cpu[i, :show], 2)
                    for i, p in enumerate(sim.profiles)
                },
                **{
                    f"{p.name} MB": np.round(sim.ram[i, :show], 0)
                    for i, p in enumerate(sim.profiles)
                },
            },
            **_WIDE,
            hide_index=True,
        )

# --- 4. Optimisation ------------------------------------------------------- #
with tabs[3]:
    st.subheader("Search progress")
    st.pyplot(fig_convergence(results))
    plt.close("all")
    st.caption(
        "The hybrid's curve is the ACO phase followed by the CSA phase, seeded "
        "with ACO's elite allocations. ACO works on a discrete grid (0.5 vCPU / "
        "128 MB steps, i.e. values VirtualBox accepts directly) which limits how "
        "close it can get; CSA and the hybrid refine continuously."
    )

    st.divider()
    st.subheader("Allocation: current vs optimised vs true demand")
    st.pyplot(fig_allocation(
        sim, problem.cur_cpu, problem.cur_ram, primary.best_cpu, primary.best_ram
    ))
    plt.close("all")

    st.divider()
    st.subheader("Algorithm comparison")
    comp = []
    for name, res in results.items():
        summary = problem.summary(res.best_cpu, res.best_ram)
        comp.append({
            "Algorithm": name,
            "Best fitness": f"{res.best_fitness:.5f}",
            "Evaluations": res.evaluations,
            "Seconds": f"{res.elapsed:.2f}",
            "Metres still unmet (vCPU)": f"{summary['cpu_unmet']:.3f}",
            "RAM still unmet (MB)": f"{summary['ram_unmet']:.0f}",
            "Under-provisioned VMs": summary["under_provisioned"],
            "Within host budget": "✅" if summary["cpu_ok"] and summary["ram_ok"] else "❌",
        })
    st.dataframe(comp, **_WIDE, hide_index=True)

    st.markdown(
        "**Reading the table** — fitness is the objective the optimiser "
        "minimises (unmet demand + waste + capacity overshoot, priority "
        "weighted). Lower is better. ACO's discrete grid leaves residual "
        "rounding error; CSA and the hybrid can match the measured demand "
        "almost exactly."
    )

    with st.expander("Optimised allocation, VM by VM"):
        st.dataframe(
            {
                "VM": sim.names,
                "Priority": sim.priorities,
                "Peak vCPU": np.round(sim.need_cpu, 2),
                "Optimised vCPU": np.round(primary.best_cpu, 3),
                "Peak MB": np.round(sim.need_ram, 0),
                "Optimised MB": np.round(primary.best_ram, 1),
            },
            **_WIDE,
            hide_index=True,
        )

# --- 5. Before vs After ---------------------------------------------------- #
with tabs[4]:
    st.subheader("Demand actually served")
    st.pyplot(fig_satisfaction(before, after, sim.names))
    plt.close("all")

    st.subheader("Stranded capacity")
    st.pyplot(fig_waste(before, after))
    plt.close("all")
    st.caption(
        "Stranded capacity is vCPU / RAM that is assigned to a VM but that the "
        "guest never uses. Before optimisation it sits on idle guests while "
        "busy guests are starved; afterwards it is reallocated."
    )

    st.divider()
    st.subheader("Full comparison table")
    st.dataframe(report.table(), **_WIDE, hide_index=True)

    with st.expander("Host level notes"):
        for note in report.host_notes:
            st.markdown(f"- {note}")

# --- 6. Recommendations ---------------------------------------------------- #
with tabs[5]:
    st.subheader("Recommended actions")
    for advice in sorted(
        report.advice,
        key=lambda a: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(a.priority, 9),
    ):
        badge = {"INCREASE": "⬆️", "REDUCE": "⬇️", "REBALANCE": "🔀", "KEEP": "✅"}[advice.action]
        with st.container(border=True):
            st.markdown(
                f"### {badge} {advice.vm} — {advice.action}\n"
                f"**Change:** {advice.delta_text()}  \n"
                f"**Priority:** `{advice.priority}`  \n"
                f"**Why:** {advice.rationale}"
            )

    st.divider()
    st.subheader("How to apply this in VirtualBox (by hand)")
    for i, step in enumerate(report.plan, start=1):
        st.markdown(f"{i}. {step}")

    st.divider()
    st.warning(report.safety, icon="🛡️")

    with st.expander("Plain-text report (copy/paste)"):
        st.code(report.summary(), language="text")

# --- 7. Host monitor ------------------------------------------------------- #
with tabs[6]:
    st.subheader("Live host monitor — read-only")
    monitor = HostMonitor(interval=0.4)

    if not monitor.available:
        st.warning(monitor.unavailable_reason())
    else:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            n_samples = st.number_input("Samples", 1, 10, 3, step=1)
            go = st.button("📸 Sample host now", type="primary", **_WIDE)
        if go:
            with st.spinner("Sampling the real machine…"):
                st.session_state["host_samples"] = monitor.sample(int(n_samples))

        samples = st.session_state.get("host_samples")
        if not samples:
            st.info("Press **Sample host now** to read the real machine with psutil. "
                    "This only reads — nothing is changed.")
        else:
            latest = samples[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CPU", f"{latest.cpu_percent:.1f} %",
                      delta=f"{latest.cpu_percent - samples[0].cpu_percent:+.1f}",
                      delta_color="inverse")
            m2.metric("RAM", f"{latest.ram_percent:.1f} %")
            m3.metric("RAM used", f"{latest.ram_used_mb:,.0f} MB",
                      delta=f"{latest.ram_used_mb - samples[0].ram_used_mb:+,.0f}",
                      delta_color="inverse")
            m4.metric("Load / core", f"{latest.load_per_core:.2f}")
            st.caption(
                f"{latest.logical_cpus} logical / {latest.core_count} physical cores · "
                f"severity {LEVEL_EMOJI.get(latest.level, '')} {latest.level} · "
                f"sampled at "
                f"{_dt.datetime.fromtimestamp(latest.timestamp):%H:%M:%S}"
            )

            if len(samples) > 1:
                fig, ax = plt.subplots(figsize=(10, 2.6))
                idx = list(range(1, len(samples) + 1))
                ax.plot(idx, [s.cpu_percent for s in samples], "o-",
                        color=PALETTE[0], label="CPU %")
                ax.plot(idx, [s.ram_percent for s in samples], "s-",
                        color=PALETTE[2], label="RAM %")
                ax.set_ylim(0, 100)
                ax.set_xlabel("sample")
                ax.set_ylabel("%")
                ax.legend(fontsize=8)
                _finish(fig, ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("**Top processes by CPU / RAM**")
            st.dataframe(
                [
                    {"Process": name, "CPU %": round(cpu, 1), "RAM (MB)": round(mem, 1)}
                    for name, cpu, mem in latest.top_processes
                ],
                **_WIDE,
                hide_index=True,
            )

    st.divider()
    st.markdown(
        "**Why this is safe:** psutil only *reads* counters. This project never "
        "calls VirtualBox, `VBoxManage`, or any process-control API, so it cannot "
        "change a VM's configuration or power state."
    )

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.divider()
st.caption(
    f"Hybrid ACO-CSA · ACO + Cuckoo Search + Streamlit · "
    f"fitness {primary.best_fitness:.5f} · {primary.evaluations} objective "
    f"evaluations in {primary.elapsed:.2f} s · {SAFETY_NOTE}"
)
