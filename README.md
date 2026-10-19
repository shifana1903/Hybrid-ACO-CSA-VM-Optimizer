# 🧮 Hybrid ACO-CSA — VirtualBox VM Resource Optimiser

Reduce the risk of **VirtualBox VM overload and crashes** by working out a better
split of the host's **CPU** and **RAM** between your virtual machines.

The project pairs **Ant Colony Optimization (ACO)** with **Cuckoo Search (CSA)**
inside a **Hybrid ACO-CSA** optimiser, wraps it in a **Streamlit** dashboard, and
explains every recommendation in plain numbers.

> ## 🛡️ Safety first
>
> * **Simulation mode only.** All numbers come from a synthetic workload model.
> * **No VirtualBox settings are ever changed.** Nothing is written to the host.
> * **No VM is ever started, stopped or restarted.** No `VBoxManage`, no bindings.
> * `psutil` is used in **read-only** mode, to show what your real machine is doing.
> * Existing files are never overwritten by the app — it only produces reports.
> * Every library used is free and open source.
>
> You read the advice, take a snapshot, and apply the change yourself.

---

## 1. The problem in one paragraph

A VirtualBox host has a fixed pool of cores and RAM. Real fleets are configured
by hand and then never revisited, so the split drifts away from reality:
a database gets 4 vCPU when it needs 5.3, while a rarely-used build agent sits on
3 vCPU it never touches. The starved guest queues work, swaps, and eventually
hangs or crashes; the over-provisioned guest just wastes capacity. This project
measures the demand, searches for a better allocation, and shows exactly which
VM should give up resources and which should get them.

---

## 2. How it works

```
        simulator.py                    monitor.py
   ┌───────────────────┐        ┌────────────────────────┐
   │ synthetic CPU/RAM │        │ psutil (read-only)     │
   │ demand traces for │        │ + RiskDetector:        │
   │ N virtual machines│        │   overload risk score  │
   └─────────┬─────────┘        └───────────┬────────────┘
             │                              │
             ▼                              ▼
   ┌──────────────────────────────────────────────────────┐
   │                    optimizer.py                      │
   │   ACO  ──► elite allocations ──►  CSA  ──► polish     │
   │  (discrete, VirtualBox-ready)   (continuous refine)   │
   └─────────────────────────┬────────────────────────────┘
                             ▼
                  recommender.py  (why + how to apply)
                             ▼
                        app.py  (Streamlit dashboard)
```

Stage by stage:

1. **Simulate** (`simulator.py`) hourly CPU (cores) and RAM (MB) demand for a
   fleet of VMs, using realistic load shapes — office-hours, 24/7, and nightly
   batch — plus noise and occasional bursts.
2. **Detect** (`monitor.py`) how the *current* allocation copes: for every VM,
   how many hours its demand exceeded its slice, how much capacity is stranded,
   and a single **0–100 overload-risk score** for the fleet.
3. **Optimise** (`optimizer.py`) the allocation with ACO, CSA, or the hybrid.
4. **Explain** (`recommender.py`) what to change, why, and how to apply it in
   VirtualBox — by hand.
5. **Visualise** (`app.py`) the whole thing, with graphs and risk alerts.

---

## 3. Project structure

| File | What it does |
|---|---|
| `app.py` | Streamlit dashboard: 7 tabs, Matplotlib graphs, risk alerts, live host monitor. |
| `simulator.py` | Synthetic VirtualBox VM workload generator (the sandbox). |
| `optimizer.py` | `ACOOptimizer`, `CSAOptimizer`, `HybridACOCSA`, the shared objective function, and the discrete/continuous local-search polish. |
| `monitor.py` | Read-only `psutil` host monitor + `RiskDetector` (risk levels, alerts, 0–100 score). |
| `recommender.py` | Recommendation engine: alerts, per-VM advice with reasons, host notes, manual application plan. |
| `requirements.txt` | The four free dependencies. |
| `README.md` | This file. |

Every module also runs standalone as a self-test (`python simulator.py`, etc.).

---

## 4. Install and run in VS Code on Windows

### 4.1 One-time setup

1. **Install Python 3.10 or newer**
   Download from <https://www.python.org/downloads/windows/>.
   On the first installer screen tick **“Add python.exe to PATH”**.

   Verify in a terminal:
   ```powershell
   python --version
   ```

2. **Install VS Code** from <https://code.visualstudio.com/>.

3. **Install the Python extension**
   Open VS Code → Extensions (`Ctrl+Shift+X`) → search **Python** (publisher:
   Microsoft, id `ms-python.python`) → **Install**.

### 4.2 Open the project

4. In VS Code: **File → Open Folder…** and choose the `Hybrid-ACO-CSA` folder.

5. Open the integrated terminal: **Terminal → New Terminal** (`Ctrl+`` ` ``).

   If you see an execution-policy error when activating the environment later,
   run this once in the same terminal:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

6. **Create and activate a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   Your prompt should now start with `(.venv)`.

7. **Tell VS Code to use it**
   `Ctrl+Shift+P` → **Python: Select Interpreter** → pick
   `./.venv/Scripts/python.exe`.

### 4.3 Install the dependencies

8. ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   That installs exactly four free packages: `streamlit`, `numpy`,
   `matplotlib`, `psutil`.

### 4.4 Run the dashboard

9. ```powershell
   python -m streamlit run app.py
   ```

   Streamlit prints a local URL — open <http://localhost:8501>.
   (If port 8501 is busy: `python -m streamlit run app.py --server.port 8502`.)

   Works from the VS Code terminal, from PowerShell, and from the **Run and
   Debug** pane (`F5`) if you add a Streamlit launch configuration.

10. Stop the server with `Ctrl+C` in the terminal.

### 4.5 Sanity-check the maths without the dashboard

Each module has a built-in demo. From the project folder:

```powershell
python simulator.py     # the synthetic fleet + its peak demand
python optimizer.py     # ACO vs CSA vs Hybrid on the same problem
python monitor.py       # live psutil snapshot + before/after risk score
python recommender.py   # the full written recommendation report
```

---

## 5. Dashboard tour

Seven tabs, all driven from the sidebar:

| Tab | What you get |
|---|---|
| **📊 Overview** | Five KPI cards, the host CPU/RAM demand curve with alarm lines, and a before/after risk comparison. |
| **⚠️ Risk alerts** | Every alert colour-coded by severity, a per-VM risk table, and an expander explaining *why* each VM was rated that way. |
| **📈 Workload** | Per-VM utilisation against the current allocation, then against the optimised one. Anything above the red 100 % line means that guest is being starved. |
| **🧠 Optimisation** | Convergence curves, the allocation bars (now / peak demand / optimised), and an algorithm comparison table. |
| **🔄 Before vs After** | Demand-served per VM, stranded capacity, the full comparison table, and host-level notes. |
| **💡 Recommendations** | One card per VM: action, exact new numbers, and the reasoning. Plus the manual VirtualBox plan and a copy/pasteable text report. |
| **🖥️ Host monitor** | Reads your real machine with `psutil` — CPU %, RAM, load per core, top processes. Read-only. |

**Sidebar controls**

* Scenario: demo fleet (6 named VMs), or a random fleet of 4 or 8 VMs.
* Simulation window in days (default 7 → 168 hourly samples).
* Trace seed and fleet seed — same seed, same trace, so results reproduce.
* Algorithm: Hybrid (recommended), ACO only, CSA only, or **Compare all three**.
* ACO ants / iterations and CSA nests / generations.
* Optimiser seed.

---

## 6. The algorithms

### 6.1 Ant Colony Optimization (`ACOOptimizer`)

ACO builds solutions the way ants find food: pheromone trails record what has
worked, and a heuristic records what looks good locally.

* **Graph.** Each VM has a discrete set of candidate CPU levels (0.5 vCPU
  steps) and RAM levels (128 MB steps) between a floor and a ceiling derived
  from its measured demand. Those are exactly the granularities VirtualBox
  accepts.
* **Construction.** Each ant visits the VMs in random order and picks a CPU
  level and a RAM level with probability
  `p ∝ τ^α · η^β`, where `τ` is the pheromone and `η` is a demand-fit
  heuristic. Levels that would bust the remaining host budget are made almost
  (not quite) impossible, so ants still explore.
* **Repair.** Any ant that overshoots the host budget is greedily downgraded
  until it fits.
* **Update.** Pheromone evaporates by `ρ`, then the iteration-best ant and the
  global-best ant deposit `Q / (fitness + f_scale)`. An elitist deposit keeps
  pressure on the best solution found.
* **Polish.** A final steepest-descent pass over discrete *transfer moves*
  (upgrade one VM, downgrade another) squeezes out the remaining rounding error
  without ever breaking the host budget.

### 6.2 Cuckoo Search (`CSAOptimizer`)

CSA searches continuously, so it is not limited by the grid.

* Each **nest** is a real-valued vector of every VM's vCPU and MB.
* New candidates are produced with a **Lévy flight** (Mantegna's algorithm,
  λ = 1.5) — mostly small steps with occasional long jumps — plus a small
  best-guided pull, and the step size **anneals** over generations.
* Every candidate is projected back into the feasible box and repaired against
  the host budget.
* Following Yang & Deb, a candidate replaces a randomly chosen nest if it is
  better, and the worst `pa = 25 %` of nests are abandoned and re-seeded each
  generation.
* A final **coordinate-descent polish** (steps shrinking from 0.25 vCPU / 64 MB
  down to 0.02 vCPU / 8 MB) tidies the result.

### 6.3 Hybrid ACO-CSA (`HybridACOCSA`)

```
ACO phase (discrete exploration, feasible by construction)
      │  elite allocations
      ▼
CSA phase (continuous refinement, Lévy flights + polish)
      │
      ▼
best of both
```

The ACO phase supplies a population that is already feasible and already close
to the demand pattern, which is exactly the kind of starting point a global
continuous search likes. The CSA phase then removes the grid rounding error.
The hybrid's convergence curve in the dashboard is the ACO curve followed by
the CSA curve, so you can see the hand-over.

### 6.4 The objective function

Every algorithm minimises the same scalar (lower is better):

| Term | Meaning |
|---|---|
| `w_short · unmet demand` | vCPU/MB the VM needs but is not given — **starvation, the crash risk**. Weighted by VM priority (critical ≫ low). |
| `w_waste · stranded capacity` | vCPU/MB assigned but never used — **inefficiency**. |
| `w_risk · under-provisioned share` | fraction of VMs left below their peak, priority weighted. |
| `w_violate · capacity overshoot` | heavily penalised; the sum of allocations must fit the host. |

Defaults: `w_short = 1.0`, `w_waste = 0.15`, `w_risk = 0.30`,
`w_violate = 15.0`. A priority-`critical` VM is punished four times as hard as
a priority-`low` VM when it is starved, which is why the optimiser will happily
shrink an idle build agent to fund a database.

---

## 7. "Risk" — what the numbers mean

`RiskDetector` replays the demand trace against an allocation and reports, per VM:

* **peak utilisation** — highest demand / slice, for CPU and RAM;
* **overloaded hours** — samples where demand exceeded the slice (the guest is
  being starved);
* **demand served** — average `min(1, slice / demand)`;
* **stranded capacity** — slice minus measured peak, floored at zero.

Severity per VM:

| Level | Rule |
|---|---|
| 🔴 `critical` | More than ~5 % of hours overloaded, or peak utilisation above 115 %. |
| 🟠 `high` | At least one overloaded hour. |
| ⚠️ `waste` | Never overloaded, but > 35 % of its slice is never used. |
| 🟢 `ok` | Sized sensibly for its measured demand. |

Fleet **risk score** (0–100):

```
score = 100 · min(1,  0.40 · under-provisioned share
                    + 0.40 · min(1, overload-sample ratio / 0.20)
                    + 0.20 · stranded-capacity share)
```

Note that this is about **starvation**, not plain utilisation: a VM sized
exactly to its measured peak legitimately runs at ~100 % at that peak. Only
demand *above* the slice counts as risk.

---

## 8. The demo scenario

`simulator.py` ships a hand-tuned 16 vCPU / 20 GB host with six VMs:

| VM | Priority | Duty | vCPU now | Peak vCPU | RAM now | Peak RAM | Verdict before |
|---|---|---|---|---|---|---|---|
| db-server | critical | 24/7 | 4.0 | 5.27 | 6144 | 7210 | 🔴 starved 153 h |
| web-server | high | office | 2.0 | 2.81 | 2048 | 2678 | 🔴 starved 130 h |
| app-server | high | office | 3.5 | 4.33 | 4096 | 5150 | 🔴 starved 98 h |
| test-vm | medium | batch | 1.0 | 1.57 | 1024 | 1643 | 🔴 starved 168 h |
| monitor-vm | medium | 24/7 | 2.5 | 0.74 | 2048 | 1236 | ⚠️ wastes 1.76 vCPU |
| build-agent | low | batch | 3.0 | 1.05 | 3072 | 1851 | ⚠️ wastes 1.95 vCPU |

The current allocation adds up to exactly 16.0 vCPU, so nothing is spare — the
capacity has to be *taken back* from the idle guests. That is the trade-off the
optimiser solves.

### Example output (seed 42)

```
Overload-risk score 75 → 0 / 100 (🔴 critical → 🟢 ok);
overloaded VM-hours 549 → 0 out of 1008.
Major improvement: the fleet moves from dangerous over-commitment to a
sustainable allocation.

RECOMMENDED CHANGES
  [CRITICAL] db-server     INCREASE  CPU 4.00 → 5.27 vCPU (+1.27) · RAM 6144 → 7214 MB (+1070)
  [HIGH    ] web-server    INCREASE  CPU 2.00 → 2.82 vCPU (+0.82) · RAM 2048 → 2682 MB (+634)
  [HIGH    ] app-server    INCREASE  CPU 3.50 → 4.33 vCPU (+0.83) · RAM 4096 → 5153 MB (+1057)
  [MEDIUM  ] test-vm       INCREASE  CPU 1.00 → 1.57 vCPU (+0.57) · RAM 1024 → 1650 MB (+626)
  [MEDIUM  ] monitor-vm    REDUCE    CPU 2.50 → 0.75 vCPU (-1.75) · RAM 2048 → 1236 MB (-812)
  [LOW     ] build-agent   REDUCE    CPU 3.00 → 1.05 vCPU (-1.95) · RAM 3072 → 1857 MB (-1215)

HOST NOTES
  - vCPU assigned: 16.00 → 15.79 of 16 (99 % of the host, +0.21 vCPU spare)
  - RAM assigned: 18432 → 19792 MB of 20480 (97 % of the host, +688 MB spare)
  - Stranded capacity falls from 3.71 vCPU / 2033 MB to 0.03 vCPU / 24 MB
```

### Typical algorithm comparison (same problem, same seed)

| Algorithm | Best fitness | Notes |
|---|---|---|
| ACO | ≈ 0.04 | Discrete 0.5 vCPU / 128 MB grid, so residual rounding error remains. Output is directly VirtualBox-compatible. |
| CSA | ≈ 0.0004 | Continuous, matches the measured demand closely. |
| **Hybrid ACO-CSA** | ≈ 0.0002 | Best final quality, seeded by ACO's elites. |

Fitness is the objective from §6.4 — lower is better. Exact values move a little
with the seed, which is normal for metaheuristics.

---

## 9. Customising

**Change the host or the fleet** — edit `DEFAULT_HOST` and `DEFAULT_PROFILES`
at the top of `simulator.py`:

```python
DEFAULT_HOST = HostSpec(total_cores=24.0, total_ram=32768)

DEFAULT_PROFILES = (
    VMProfile("db-server", "Ubuntu 22.04", 6.0, 8192, 7.5, 9000, "critical", "always"),
    ...
)
```

**Change the trade-off** — the weights live on `AllocationProblem`
(`w_short`, `w_waste`, `w_risk`, `w_violate`). Raise `w_waste` if you care more
about efficiency than about guaranteeing peak demand; raise `w_risk` if you want
the optimiser to refuse to leave *any* VM under-provisioned.

**Change the search effort** — the sidebar sliders, or the constructor
arguments:

```python
HybridACOCSA(aco_ants=30, aco_iterations=80, csa_nests=40, csa_generations=300)
```

**Feed in real measurements** — everything downstream of `simulator.py` only
needs a `SimulationResult` with `cpu` and `ram` arrays of shape
`(n_vms, n_samples)`. Record real per-VM CPU/RAM with `psutil` or VirtualBox's
own metrics and build the same object; the optimizer, risk detector and
recommender will work unchanged.

---

## 10. Troubleshooting (Windows)

| Symptom | Fix |
|---|---|
| `streamlit : The term 'streamlit' is not recognized` | Use `python -m streamlit run app.py`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `python` opens the Microsoft Store | Re-run the Python installer → *Modify* → tick **Add python.exe to PATH**; or use the `py` launcher: `py -m streamlit run app.py`. |
| Port 8501 already in use | `python -m streamlit run app.py --server.port 8502`. |
| `UnicodeEncodeError` when running a module's self-test | `set PYTHONUTF8=1` first, or run `chcp 65001` once in that terminal. |
| Optimisation feels slow | Lower *ACO iterations* / *CSA generations* in the sidebar. The dashboard caches results, so only the first run of a given setting is slow. |
| `ModuleNotFoundError: No module named 'streamlit'` | Make sure the `(.venv)` prompt is showing, then `pip install -r requirements.txt`. |
| Blank page / stuck spinner | Hard-refresh the browser (`Ctrl+F5`) and check the terminal for a traceback. |

---

## 11. Limitations & disclaimer

* The workload model is **synthetic**. It is realistic in shape (diurnal peaks,
  weekday/weekend effects, nightly batch, bursts, sticky RAM) but it is not
  your workload. Use it to compare *allocations*, not to predict exact numbers.
* VirtualBox over-commits CPU by design, and vCPU counts interact with host
  scheduling, NUMA, nested paging and guest OS behaviour in ways a simple model
  cannot capture. Treat the output as a strong hint, not a guarantee.
* RAM allocation in VirtualBox is the one setting where over-committing really
  bites; the recommender deliberately warns when host RAM headroom drops below
  1 GB.
* This tool is **advisory**. It never applies anything. Take a snapshot, change
  one VM, watch it for a day, then do the next one.

---

## 12. Built with

`Python 3.10+` · `streamlit` · `numpy` · `matplotlib` · `psutil` — all free and
open source. No paid services, no cloud accounts, no telemetry.
