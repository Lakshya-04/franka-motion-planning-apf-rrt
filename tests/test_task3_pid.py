"""
Task 3 Validation — Haptic Dial PID
=====================================
Runs two side-by-side simulations in Python (replicating the C++ logic) and
produces a matplotlib figure showing:

  Left  — WITHOUT anti-windup: integral winds up at the end-stop, then
           dumps a large torque spike when the stop is released.
  Right — WITH anti-windup (clamping method, as in haptic_pid.cpp): integral
           is frozen at saturation; no lurch on release.

Additionally shows the D-term low-pass filter effect in a third panel.

Run:
    python tests/test_task3_pid.py
    # or from project root:
    make test-task3
"""

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless — saves PNG without needing a display
import matplotlib.pyplot as plt

# Output directory sits next to this file: tests/results/
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# PID implementation (mirrors haptic_pid.cpp exactly)
# ---------------------------------------------------------------------------


class PID:
    def __init__(self, kp, ki, kd, dt, out_min, out_max, lpf_hz=50.0, anti_windup=True):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.out_min, self.out_max = out_min, out_max
        self.anti_windup = anti_windup
        alpha_denom = dt + 1.0 / (2.0 * math.pi * lpf_hz)
        self.lpf_alpha = dt / alpha_denom
        self.integral = 0.0
        self.prev_error = 0.0
        self.d_filtered = 0.0

    def update(self, setpoint, measured):
        error = setpoint - measured
        p_term = self.kp * error

        d_raw = (error - self.prev_error) / self.dt
        self.d_filtered = self.lpf_alpha * d_raw + (1.0 - self.lpf_alpha) * self.d_filtered
        d_term = self.kd * self.d_filtered
        self.prev_error = error

        output_pd = p_term + d_term

        if self.anti_windup:
            i_cand = self.integral + self.ki * error * self.dt
            lo = self.out_min - output_pd
            hi = self.out_max - output_pd
            self.integral = max(lo, min(hi, i_cand))
        else:
            self.integral += self.ki * error * self.dt

        output = output_pd + self.integral
        return max(self.out_min, min(self.out_max, output))


# ---------------------------------------------------------------------------
# Plant + scenario
# ---------------------------------------------------------------------------


def nearest_detent(angle, detents=24):
    step = 2.0 * math.pi / detents
    return round(angle / step) * step


def run_sim(anti_windup: bool):
    DT = 0.001
    SIM_SECONDS = 4.0
    N = int(SIM_SECONDS / DT)
    DETENTS = 24
    MOTOR_GAIN = 6.0
    DAMPING = 0.98
    BACKLASH = math.radians(0.5)

    sp_a = nearest_detent(2 * 2 * math.pi / DETENTS)
    sp_b = nearest_detent(5 * 2 * math.pi / DETENTS)
    stop_angle = sp_b + math.radians(30)

    pid = PID(
        kp=3.5, ki=0.8, kd=0.12, dt=DT,
        out_min=-1.0, out_max=1.0,
        lpf_hz=50.0, anti_windup=anti_windup,
    )

    true_pos = 0.0
    velocity = 0.0

    t_arr = np.empty(N)
    sp_arr = np.empty(N)
    pos_arr = np.empty(N)
    out_arr = np.empty(N)
    int_arr = np.empty(N)

    for k in range(N):
        t = k * DT

        if t < 1.0:
            setpoint = sp_a
        elif t < 2.0:
            setpoint = sp_b
        elif t < 3.0:
            setpoint = stop_angle + math.radians(15)
        else:
            setpoint = sp_b

        measured = true_pos
        output = pid.update(setpoint, measured)

        velocity += output * MOTOR_GAIN * DT
        velocity *= DAMPING
        candidate = true_pos + velocity * DT

        if 2.0 <= t < 3.0 and candidate >= stop_angle:
            true_pos = stop_angle
            velocity = 0.0
        else:
            true_pos = candidate

        t_arr[k] = t
        sp_arr[k] = setpoint
        pos_arr[k] = true_pos
        out_arr[k] = output
        int_arr[k] = pid.integral

    return t_arr, sp_arr, pos_arr, out_arr, int_arr


# ---------------------------------------------------------------------------
# D-term LPF demo (noisy derivative, filtered vs unfiltered)
# ---------------------------------------------------------------------------


def run_lpf_demo():
    DT = 0.001
    N = 1000
    t = np.arange(N) * DT
    true_signal = np.where(t < 0.2, 0.0, 1.0)
    noise = np.random.default_rng(42).normal(0, 0.003, N)
    measured = true_signal + noise

    d_raw = np.diff(measured, prepend=measured[0]) / DT
    alpha = DT / (DT + 1.0 / (2 * math.pi * 50.0))
    d_filtered = np.zeros(N)
    for i in range(1, N):
        d_filtered[i] = alpha * d_raw[i] + (1 - alpha) * d_filtered[i - 1]

    return t, d_raw, d_filtered


# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------


def main():
    t_no, sp_no, pos_no, out_no, int_no = run_sim(anti_windup=False)
    t_aw, sp_aw, pos_aw, out_aw, int_aw = run_sim(anti_windup=True)
    t_lpf, d_raw, d_filt = run_lpf_demo()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Task 3 — Haptic Dial PID Validation", fontsize=14, fontweight="bold")

    shade_kw = dict(alpha=0.12, color="red")

    for col, (t, sp, pos, out, integral, title) in enumerate([
        (t_no, sp_no, pos_no, out_no, int_no, "WITHOUT anti-windup"),
        (t_aw, sp_aw, pos_aw, out_aw, int_aw, "WITH anti-windup (clamping)"),
    ]):
        ax_pos = axes[0, col]
        ax_int = axes[1, col]

        ax_pos.plot(t, sp, "k--", lw=1.2, label="setpoint")
        ax_pos.plot(t, pos, "b", lw=1.5, label="position")
        ax_pos.axvspan(2.0, 3.0, **shade_kw, label="at end-stop")
        ax_pos.set_title(title, fontsize=11)
        ax_pos.set_ylabel("angle (rad)")
        ax_pos.legend(fontsize=8)
        ax_pos.set_xlim(0, 4)
        ax_pos.grid(True, alpha=0.3)

        ax_int.plot(t, integral, "g", lw=1.5, label="integral term")
        ax_int.plot(t, out, "r", lw=1.0, alpha=0.6, label="output")
        ax_int.axvspan(2.0, 3.0, **shade_kw, label="at end-stop")
        ax_int.set_xlabel("time (s)")
        ax_int.set_ylabel("value")
        ax_int.set_xlim(0, 4)
        ax_int.legend(fontsize=8)
        ax_int.grid(True, alpha=0.3)

    ax_lpf_raw = axes[0, 2]
    ax_lpf_filt = axes[1, 2]

    ax_lpf_raw.plot(t_lpf, d_raw, "r", lw=0.8, alpha=0.7)
    ax_lpf_raw.set_title("D-term: raw (unfiltered)", fontsize=11)
    ax_lpf_raw.set_ylabel("d/dt measured")
    ax_lpf_raw.set_ylim(-300, 300)
    ax_lpf_raw.grid(True, alpha=0.3)

    ax_lpf_filt.plot(t_lpf, d_filt, "b", lw=1.5)
    ax_lpf_filt.set_title("D-term: after 50 Hz LPF", fontsize=11)
    ax_lpf_filt.set_xlabel("time (s)")
    ax_lpf_filt.set_ylabel("d/dt measured")
    ax_lpf_filt.set_ylim(-300, 300)
    ax_lpf_filt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "test_task3_pid.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
