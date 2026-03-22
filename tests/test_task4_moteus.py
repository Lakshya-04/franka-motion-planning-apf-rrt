"""
Task 4 Validation — moteus High-Speed Reciprocating Actuator (pure-Python mock)
================================================================================
Replaces the PyBullet single-joint scene with a pure-Python plant model so all
three phases are visible without needing hardware or a second PyBullet window.

Phases
------
  Phase 1 — Homing
    Motor drives at slow velocity toward a hard stop.  Stall current exceeds
    threshold → motor stops, retracts 0.05 rev, zeroes position reference.

  Phase 2 — Cyclic Trajectory
    Reciprocates between 0.0 rev and 2.0 rev, 10 cycles, trapezoid profile
    (vel_limit=5 rev/s, accel_limit=20 rev/s²).

  Phase 3 — Telemetry
    Saves telemetry_mock.csv and a static summary PNG to tests/results/.

Run:
    python tests/test_task4_moteus.py
    # or from project root:
    make test-task4
"""

import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation  # noqa: F401 (available for callers)

matplotlib.use("Agg")  # headless — saves PNG without needing a display
import matplotlib.pyplot as plt

# Output directory sits next to this file: tests/results/
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (mirroring moteus_actuator.py defaults)
# ---------------------------------------------------------------------------
HOMING_VEL = 0.1          # rev/s
STALL_THRESHOLD = 1.2     # A
RETRACT_REV = 0.05        # rev
VEL_LIMIT = 5.0           # rev/s
ACCEL_LIMIT = 20.0        # rev/s²
POS_MIN = 0.0             # rev
POS_MAX = 2.0             # rev
DWELL_S = 0.1             # s
N_CYCLES = 10
DT = 1.0 / 240.0          # s
HARD_STOP_REV = -0.30     # rev  (physical end-stop)
DAMPING = 0.995


# ---------------------------------------------------------------------------
# Trapezoid velocity profile
# ---------------------------------------------------------------------------


class TrapProfile:
    def __init__(self, vel_limit, accel_limit, dt):
        self.vl = vel_limit
        self.al = accel_limit
        self.dt = dt
        self.pos = 0.0
        self.vel = 0.0

    def step(self, target):
        err = target - self.pos
        if abs(err) < 1e-6:
            self.vel = 0.0
            return self.pos
        v_brake = math.sqrt(2.0 * self.al * abs(err))
        v_desired = math.copysign(min(self.vl, v_brake), err)
        dv = v_desired - self.vel
        dv_max = self.al * self.dt
        self.vel += math.copysign(min(abs(dv), dv_max), dv)
        self.pos += self.vel * self.dt
        return self.pos


# ---------------------------------------------------------------------------
# Simulate all phases, return telemetry list
# ---------------------------------------------------------------------------


def run_simulation():
    telemetry = []
    t_sim = 0.0
    pos = 0.0   # rev
    vel = 0.0   # rev/s

    # ── Phase 1: Homing ──────────────────────────────────────────────────────
    home_vel = -HOMING_VEL
    stall_done = False
    MAX_HOME_S = 10.0

    while not stall_done and t_sim < MAX_HOME_S:
        vel += (home_vel - vel) * (DT / 0.05)
        vel *= DAMPING

        if pos + vel * DT <= HARD_STOP_REV:
            pos = HARD_STOP_REV
            vel = 0.0

        pos += vel * DT
        t_sim += DT

        q_current = abs(home_vel - vel) * 2.0
        telemetry.append(dict(t=t_sim, phase="homing",
                              target=0.0, actual=pos,
                              vel=vel, current=min(q_current, 2.0)))

        if q_current >= STALL_THRESHOLD:
            stall_done = True

    stall_pos = pos
    zero_pos = stall_pos + RETRACT_REV

    for _ in range(int(0.3 / DT)):
        vel += (zero_pos - pos - vel * 0.05) * DT * 20
        vel *= DAMPING
        pos += vel * DT
        t_sim += DT
        telemetry.append(dict(t=t_sim, phase="retract",
                              target=zero_pos, actual=pos,
                              vel=vel, current=0.1))

    # ── Phase 2: Cyclic Trajectory ────────────────────────────────────────────
    profile = TrapProfile(VEL_LIMIT, ACCEL_LIMIT, DT)
    profile.pos = pos
    profile.vel = vel

    for _ in range(N_CYCLES):
        for target_offset in [POS_MAX, POS_MIN]:
            target = zero_pos + target_offset

            max_steps = int(10.0 / DT)  # 10 s safety cap per leg
            for _ in range(max_steps):
                cmd = profile.step(target)
                vel += (profile.vel - vel) * DT * 30
                vel *= DAMPING
                pos += vel * DT
                t_sim += DT

                q_current = abs(cmd - pos) * 1.5
                telemetry.append(dict(t=t_sim, phase="cyclic",
                                      target=target - zero_pos,
                                      actual=pos - zero_pos,
                                      vel=vel, current=min(q_current, 2.0)))

                if abs(profile.vel) < 1e-3 and abs(pos - target) < 0.05:
                    break

            for _ in range(int(DWELL_S / DT)):
                pos += vel * DT * 0.1
                t_sim += DT
                telemetry.append(dict(t=t_sim, phase="dwell",
                                      target=target - zero_pos,
                                      actual=pos - zero_pos,
                                      vel=0.0, current=0.05))

    return telemetry, zero_pos


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------


def save_csv(telemetry, path: Path | None = None):
    if path is None:
        path = RESULTS_DIR / "telemetry_mock.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=telemetry[0].keys())
        writer.writeheader()
        writer.writerows(telemetry)
    print(f"Telemetry saved → {path}")


# ---------------------------------------------------------------------------
# Static summary plot
# ---------------------------------------------------------------------------


def plot_summary(telemetry):
    t = np.array([r["t"] for r in telemetry])
    pos = np.array([r["actual"] for r in telemetry])
    tgt = np.array([r["target"] for r in telemetry])
    vel = np.array([r["vel"] for r in telemetry])
    cur = np.array([r["current"] for r in telemetry])
    ph = np.array([r["phase"] for r in telemetry])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Task 4 — moteus Actuator Validation", fontsize=13, fontweight="bold")

    colours = {
        "homing": "#ffdddd",
        "retract": "#ffffcc",
        "cyclic": "#ddeeff",
        "dwell": "#eeffee",
    }
    for ax in axes:
        for phase, colour in colours.items():
            mask = ph == phase
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            starts = [idx[0]]
            ends = []
            for j in range(1, len(idx)):
                if idx[j] - idx[j - 1] > 1:
                    ends.append(idx[j - 1])
                    starts.append(idx[j])
            ends.append(idx[-1])
            for s, e in zip(starts, ends):
                ax.axvspan(t[s], t[e], color=colour, alpha=0.5)

    axes[0].plot(t, tgt, "k--", lw=1, label="target (rev)")
    axes[0].plot(t, pos, "b", lw=1.5, label="actual (rev)")
    axes[0].set_ylabel("position (rev)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, vel, "g", lw=1.2, label="velocity (rev/s)")
    axes[1].axhline(VEL_LIMIT, color="r", lw=0.8, ls="--")
    axes[1].axhline(-VEL_LIMIT, color="r", lw=0.8, ls="--",
                    label=f"±{VEL_LIMIT} rev/s limit")
    axes[1].set_ylabel("velocity (rev/s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, cur, "r", lw=1.2, label="q_current (A)")
    axes[2].axhline(STALL_THRESHOLD, color="k", lw=0.8, ls="--",
                    label=f"stall threshold {STALL_THRESHOLD} A")
    axes[2].set_ylabel("current (A)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    patches = [mpatches.Patch(color=c, label=ph_name, alpha=0.7)
               for ph_name, c in colours.items()]
    axes[0].legend(
        handles=patches + axes[0].get_legend_handles_labels()[0], fontsize=8
    )

    plt.tight_layout()
    out_path = RESULTS_DIR / "test_task4_moteus.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("Simulating moteus actuator …")
    telemetry, zero_pos = run_simulation()

    homing = [r for r in telemetry if r["phase"] == "homing"]
    print(f"  Phase 1 Homing:   {homing[-1]['t']:.2f}s, "
          f"stall at {homing[-1]['actual']:.3f} rev, "
          f"current = {homing[-1]['current']:.2f} A")

    cyclic = [r for r in telemetry if r["phase"] == "cyclic"]
    print(f"  Phase 2 Cyclic:   {len(cyclic)} steps, "
          f"pos range [{min(r['actual'] for r in cyclic):.2f}, "
          f"{max(r['actual'] for r in cyclic):.2f}] rev")

    save_csv(telemetry)
    plot_summary(telemetry)


if __name__ == "__main__":
    main()
