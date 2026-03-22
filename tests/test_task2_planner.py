"""
Task 2 Validation — Hybrid APF-Guided RRT Motion Planner
=========================================================
Validates three layers:

  1. Unit checks (pure math, no PyBullet)
       - APFGradient.attractive()
       - RRTPlanner.path_length()
       - RRTPlanner._nearest()
       - RRTPlanner._extend() in vanilla mode

  2. Integration (PyBullet DIRECT, N runs each)
       - Vanilla RRT finds a collision-free path Q_START → Q_GOAL
       - APF-RRT finds a path faster / with fewer nodes
       - PSO smoothing reduces total path length

  3. Path quality metrics (averaged over N runs)
       - path length      (rad)          — total joint-space arc length
       - smoothness       (rad/step)     — max angular step between waypoints (lower = smoother)
       - efficiency       (ratio)        — path length / straight-line distance (1.0 = perfectly direct)
       - goal error       (rad)          — ||last_waypoint − Q_GOAL|| (should be ≈ 0)
       - clearance        (m)            — minimum obstacle clearance along path
       - success rate     (%)            — fraction of runs that found a path

  4. Per-run CSV saved to tests/results/test_task2_runs.csv
  5. Summary plot saved to tests/results/test_task2_planner.png

Run:
    python tests/test_task2_planner.py          # default 5 runs
    python tests/test_task2_planner.py --runs 10
    # or from project root:
    make test-task2
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure the task2 package is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from task2_motion_planning.apf_rrt_planner import (
    APFGradient,
    JOINT_LOWER,
    JOINT_UPPER,
    Node,
    NUM_JOINTS,
    OBSTACLES,
    PSOPathSmoother,
    Q_GOAL,
    Q_START,
    RobotEnv,
    RRTPlanner,
    plan_parallel_portfolio,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Straight-line joint-space distance from start to goal (theoretical minimum)
_STRAIGHT_LINE = float(np.linalg.norm(Q_GOAL - Q_START))


# ---------------------------------------------------------------------------
# Path quality helpers
# ---------------------------------------------------------------------------


def path_smoothness(path: List[np.ndarray]) -> float:
    """Max joint-space step between consecutive waypoints (lower = smoother)."""
    if len(path) < 2:
        return 0.0
    return float(max(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def path_efficiency(path: List[np.ndarray]) -> float:
    """Ratio of straight-line distance to actual path length (1.0 = perfectly direct)."""
    length = RRTPlanner.path_length(path)
    return _STRAIGHT_LINE / length if length > 0 else 0.0


def goal_error(path: List[np.ndarray]) -> float:
    """Joint-space distance from the last waypoint to Q_GOAL."""
    return float(np.linalg.norm(path[-1] - Q_GOAL))


def path_clearance(path: List[np.ndarray], env: RobotEnv, sample_every: int = 10) -> float:
    """
    Minimum Euclidean clearance (m) between any link origin and any obstacle
    surface along the path. Samples every `sample_every` waypoints for speed.
    """
    min_clearance = float("inf")
    for q in path[::sample_every]:
        link_positions = env.get_link_positions(q)
        for lp in link_positions:
            for obs_centre, obs_radius in OBSTACLES:
                d = np.linalg.norm(lp - np.array(obs_centre)) - obs_radius
                min_clearance = min(min_clearance, d)
    return min_clearance


def compute_metrics(path: List[np.ndarray], env: RobotEnv, plan_time: float) -> dict:
    return {
        "length":     RRTPlanner.path_length(path),
        "smoothness": path_smoothness(path),
        "efficiency": path_efficiency(path),
        "goal_error": goal_error(path),
        "clearance":  path_clearance(path, env),
        "time":       plan_time,
        "waypoints":  len(path),
    }


# ---------------------------------------------------------------------------
# 1. Unit checks (no PyBullet)
# ---------------------------------------------------------------------------


def test_attractive():
    """attractive() returns k_att * (q_rand - q) without touching PyBullet."""
    apf = APFGradient.__new__(APFGradient)
    apf.k_att = 2.0

    q = np.zeros(NUM_JOINTS)
    q_rand = np.ones(NUM_JOINTS)
    result = apf.attractive(q, q_rand)
    expected = 2.0 * np.ones(NUM_JOINTS)
    assert np.allclose(result, expected), f"attractive() wrong: {result}"
    print("  [PASS] APFGradient.attractive()")


def test_path_length():
    """path_length() sums Euclidean distances between consecutive waypoints."""
    p0 = np.zeros(NUM_JOINTS)
    p1 = np.ones(NUM_JOINTS)
    p2 = 2.0 * np.ones(NUM_JOINTS)
    length = RRTPlanner.path_length([p0, p1, p2])
    expected = 2.0 * np.sqrt(NUM_JOINTS)
    assert abs(length - expected) < 1e-9, f"path_length() wrong: {length}"
    print("  [PASS] RRTPlanner.path_length()")


def test_nearest():
    """_nearest() returns the closest node by joint-space distance."""
    planner = RRTPlanner.__new__(RRTPlanner)
    planner.tree = [
        Node(q=np.zeros(NUM_JOINTS)),
        Node(q=np.ones(NUM_JOINTS)),
        Node(q=np.full(NUM_JOINTS, 0.1)),
    ]
    q_rand = np.full(NUM_JOINTS, 0.05)
    idx, q_near = planner._nearest(q_rand)
    assert idx == 0, f"_nearest() returned wrong index: {idx}"
    assert np.allclose(q_near, np.zeros(NUM_JOINTS))
    print("  [PASS] RRTPlanner._nearest()")


def test_extend_vanilla():
    """_extend() (vanilla, no APF) steps step_size toward q_rand."""
    planner = RRTPlanner.__new__(RRTPlanner)
    planner.apf = None
    planner.step_size = 0.3

    q_near = Q_START.copy()
    direction = np.ones(NUM_JOINTS) / np.sqrt(NUM_JOINTS)
    q_rand = q_near + 2.0 * direction
    q_new = planner._extend(q_near, q_rand)
    assert q_new is not None
    dist = np.linalg.norm(q_new - q_near)
    assert abs(dist - 0.3) < 1e-6, f"_extend() step size wrong: {dist}"
    assert np.all(q_new >= JOINT_LOWER) and np.all(q_new <= JOINT_UPPER), \
        "_extend() violated joint limits"
    print("  [PASS] RRTPlanner._extend() vanilla")


def test_path_smoothness_metric():
    """smoothness() returns the largest step, efficiency() stays ≤ 1.0."""
    straight = [Q_START, Q_GOAL]
    assert abs(path_smoothness(straight) - _STRAIGHT_LINE) < 1e-9
    assert abs(path_efficiency(straight) - 1.0) < 1e-9
    assert goal_error(straight) < 1e-9
    print("  [PASS] quality metric helpers")


# ---------------------------------------------------------------------------
# 2. Integration (PyBullet DIRECT, N runs)
# ---------------------------------------------------------------------------


_MAX_ITER = 4000   # per-trial cap for single-process planners
_N_WORKERS = 4     # parallel portfolio workers


def _make_metrics_nan():
    m = {k: float("nan") for k in ("length", "smoothness", "efficiency",
                                    "goal_error", "clearance", "time", "waypoints")}
    m["nodes"] = float("nan")
    m["success"] = 0
    return m


def _run_once(env: RobotEnv, pso: PSOPathSmoother) -> dict:
    """
    One trial — four planners compared:
      vanilla     : single-tree RRT, no APF
      apf         : single-tree APF-RRT (stall detection + adaptive bias)
      bidir       : bidirectional RRT-Connect + APF
      parallel    : parallel portfolio (N workers × bidirectional APF-RRT)
    PSO smoothing applied to whichever of apf / bidir / parallel succeeds first.
    """
    row = {}

    # ── Vanilla RRT ────────────────────────────────────────────────────
    t0 = time.time()
    v_planner = RRTPlanner(env, apf=None, max_iter=_MAX_ITER)
    v_path = v_planner.plan(Q_START, Q_GOAL)
    v_time = time.time() - t0
    if v_path is not None:
        m = compute_metrics(v_path, env, v_time)
        m["nodes"] = v_planner.stats["node_count"]
        m["success"] = 1
    else:
        m = _make_metrics_nan()
        m["nodes"] = v_planner.stats["node_count"]
    row["vanilla"] = m

    # ── APF-RRT (single-tree) ──────────────────────────────────────────
    apf = APFGradient(env)
    t0 = time.time()
    a_planner = RRTPlanner(env, apf=apf, max_iter=_MAX_ITER)
    a_path = a_planner.plan(Q_START, Q_GOAL)
    a_time = time.time() - t0
    if a_path is not None:
        m = compute_metrics(a_path, env, a_time)
        m["nodes"] = a_planner.stats["node_count"]
        m["success"] = 1
    else:
        m = _make_metrics_nan()
        m["nodes"] = a_planner.stats["node_count"]
    row["apf"] = m

    # ── Bidirectional APF-RRT ─────────────────────────────────────────
    t0 = time.time()
    b_planner = RRTPlanner(env, apf=APFGradient(env), max_iter=_MAX_ITER)
    b_path = b_planner.plan_bidirectional(Q_START, Q_GOAL)
    b_time = time.time() - t0
    if b_path is not None:
        m = compute_metrics(b_path, env, b_time)
        m["nodes"] = b_planner.stats["node_count"]
        m["success"] = 1
    else:
        m = _make_metrics_nan()
        m["nodes"] = b_planner.stats["node_count"]
    row["bidir"] = m

    # ── Parallel portfolio (N workers, bidirectional APF-RRT) ─────────
    p_path, p_stats = plan_parallel_portfolio(
        n_workers=_N_WORKERS, use_apf=True, bidirectional=True, max_iter=_MAX_ITER
    )
    if p_path is not None:
        m = compute_metrics(p_path, env, p_stats["total_time"])
        m["nodes"] = p_stats.get("node_count", float("nan"))
        m["success"] = 1
    else:
        m = _make_metrics_nan()
    row["parallel"] = m

    # ── Shortcut + PSO smoothing on the best successful path ──────────
    best_path = next((r for r in [p_path, b_path, a_path] if r is not None), None)
    best_time = next((row[k]["time"] for k in ["parallel", "bidir", "apf"]
                      if row[k]["success"]), float("nan"))
    if best_path is not None:
        t0 = time.time()
        # Greedy shortcutting first: collapses redundant bidir waypoints cheaply,
        # giving PSO a much shorter starting path to optimise from.
        sc_planner = RRTPlanner(env, apf=None, max_iter=1)
        sc_path = sc_planner.shortcut_path(best_path)
        s_path = pso.smooth(sc_path)
        s_time = time.time() - t0
        sm = compute_metrics(s_path, env, best_time + s_time)
        sm["nodes"] = float("nan")
        sm["success"] = 1
    else:
        sm = _make_metrics_nan()
    row["pso"] = sm

    return row


def _nanmean(values):
    arr = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(arr)) if arr else float("nan")


def run_integration(n_runs: int = 5):
    env = RobotEnv(use_gui=False)
    pso = PSOPathSmoother(env, n_particles=20, n_iter=100)

    all_rows = []
    csv_rows = []

    print(f"  Running {n_runs} trials  [{_N_WORKERS} parallel workers per portfolio run] …")
    for i in range(n_runs):
        row = _run_once(env, pso)
        all_rows.append(row)
        v, a, b, p, s = row["vanilla"], row["apf"], row["bidir"], row["parallel"], row["pso"]
        print(f"  run {i+1:2d} | "
              f"vanilla: {'OK' if v['success'] else 'FAIL'} t={v['time']:.2f}s | "
              f"apf: {'OK' if a['success'] else 'FAIL'} t={a['time']:.2f}s | "
              f"bidir: {'OK' if b['success'] else 'FAIL'} t={b['time']:.2f}s | "
              f"parallel: {'OK' if p['success'] else 'FAIL'} t={p['time']:.2f}s | "
              f"pso: len={s['length']:.2f}")
        csv_rows.append({
            "run":                i + 1,
            "vanilla_success":    v["success"],
            "vanilla_length":     round(v["length"],     4),
            "vanilla_nodes":      v["nodes"],
            "vanilla_time":       round(v["time"],       3),
            "apf_success":        a["success"],
            "apf_length":         round(a["length"],     4),
            "apf_nodes":          a["nodes"],
            "apf_time":           round(a["time"],       3),
            "bidir_success":      b["success"],
            "bidir_length":       round(b["length"],     4),
            "bidir_nodes":        b["nodes"],
            "bidir_time":         round(b["time"],       3),
            "parallel_success":   p["success"],
            "parallel_length":    round(p["length"],     4),
            "parallel_time":      round(p["time"],       3),
            "pso_success":        s["success"],
            "pso_length":         round(s["length"],     4),
            "pso_time":           round(s["time"],       3),
        })

    # Save CSV
    csv_path = RESULTS_DIR / "test_task2_runs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  CSV  → {csv_path}")

    # Compute averages (ignoring failed runs)
    def avg(planner, key):
        return _nanmean([r[planner][key] for r in all_rows])

    summary = {}
    for planner in ("vanilla", "apf", "bidir", "parallel", "pso"):
        summary[planner] = {
            "length":     avg(planner, "length"),
            "nodes":      avg(planner, "nodes"),
            "smoothness": avg(planner, "smoothness"),
            "efficiency": avg(planner, "efficiency"),
            "clearance":  avg(planner, "clearance"),
            "time":       avg(planner, "time"),
            "success":    _nanmean([r[planner]["success"] for r in all_rows]) * 100,
        }

    print(f"\n  {'Planner':<18} {'success':>8} {'length':>8} {'nodes':>7} "
          f"{'smooth':>8} {'eff':>6} {'clear':>8} {'time':>7}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*8} {'-'*7}")
    for label, key in [("Vanilla RRT", "vanilla"), ("APF-RRT", "apf"),
                       ("Bidir APF-RRT", "bidir"), ("Parallel", "parallel"),
                       ("APF+PSO", "pso")]:
        s = summary[key]
        print(f"  {label:<18} {s['success']:>7.0f}% {s['length']:>8.3f} {s['nodes']:>7.0f} "
              f"{s['smoothness']:>8.3f} {s['efficiency']:>6.2f} "
              f"{s['clearance']:>8.3f} {s['time']:>7.2f}s")

    return summary


# ---------------------------------------------------------------------------
# 3. Summary plot
# ---------------------------------------------------------------------------


def plot_summary(results, n_runs: int):
    labels  = ["Vanilla\nRRT", "APF-RRT", "Bidir\nAPF-RRT", "Parallel\nPortfolio", "APF-RRT\n+PSO"]
    keys    = ["vanilla",      "apf",     "bidir",           "parallel",            "pso"]
    colours = ["#5588cc",      "#ee8833", "#cc44aa",         "#cc3333",             "#55aa55"]

    metrics = [
        ("success",    "success rate (%)",        ".0f"),
        ("length",     "avg path length (rad)",   ".2f"),
        ("nodes",      "avg tree nodes",           ".0f"),
        ("time",       "avg compute time (s)",    ".2f"),
        ("smoothness", "avg max step (rad)",       ".3f"),
        ("efficiency", "avg path efficiency",     ".2f"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Task 2 — APF-Guided RRT Validation ({n_runs} runs, averages)",
                 fontsize=13, fontweight="bold")

    for ax, (metric_key, ylabel, fmt) in zip(axes.flat, metrics):
        values = [results[k][metric_key] for k in keys]
        valid = [v for v in values if not np.isnan(v)]
        top = max(valid) * 1.30 if valid and max(valid) > 0 else 1.0
        bars = ax.bar(labels, values, color=colours, width=0.55)
        ax.set_title(ylabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, top)
        for bar, val in zip(bars, values):
            label_txt = format(val, fmt) if not np.isnan(val) else "n/a"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.02,
                    label_txt, ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / "test_task2_planner.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of planning trials per planner (default: 5)")
    args = parser.parse_args()

    print("── Unit checks (pure math) ──────────────────────────────────────")
    test_attractive()
    test_path_length()
    test_nearest()
    test_extend_vanilla()
    test_path_smoothness_metric()

    print(f"── Integration (PyBullet DIRECT, {args.runs} runs) ──────────────────────")
    results = run_integration(n_runs=args.runs)

    plot_summary(results, n_runs=args.runs)


if __name__ == "__main__":
    main()
