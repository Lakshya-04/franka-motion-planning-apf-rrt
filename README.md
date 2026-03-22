# SocialHW Robotics Assessment — Full Solution

Four technical tasks for the **Robotics Engineer** position, implemented in Python / C++.

---

## Repository Layout

```
social-hw-task/
├── Makefile
├── requirements.txt
├── task1_perception/          # Task 1 — Pick-and-Place pipeline
│   ├── pipeline.py
│   ├── detection.py
│   ├── controller.py
│   ├── scene.py
│   ├── camera.py
│   ├── config.py
│   └── results/
├── task2_motion_planning/     # Task 2 — Hybrid APF-Guided RRT
│   ├── apf_rrt_planner.py
│   └── results/
├── task3_haptic_pid/          # Task 3 — Haptic Dial PID (C++)
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── haptic_pid.cpp
│   │   └── main.cpp
│   └── include/
│       └── haptic_pid.hpp
├── task4_moteus/              # Task 4 — High-Speed Reciprocating Actuator
│   ├── moteus_actuator.py
│   └── results/
└── tests/                     # Validation test suite
    ├── test_task3_pid.py
    ├── test_task4_moteus.py
    └── results/               # Generated PNGs and CSVs
```

---

## Quick Start

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Task 4 additionally requires the `moteus` package (hardware only):

```bash
pip install moteus
```

### 2. Build, lint, and test everything

```bash
make all
```

Or individually:

```bash
make build        # compile Task 3 C++ binary via CMake
make test         # run Task 2 + Task 3 + Task 4 validation scripts
make lint         # pylint over all Python packages
make clean        # remove build artifacts and __pycache__
```

### 3. Run individual tests

```bash
make test-task2   # Task 2 planner validation → tests/results/test_task2_planner.png
make test-task3   # Task 3 PID validation → tests/results/test_task3_pid.png
make test-task4   # Task 4 moteus validation → tests/results/test_task4_moteus.png
```

### Prerequisites

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| cmake | >= 3.16 |
| g++ | C++17 support |

---

## Task 1 — Perception & Control

**Package:** `task1_perception/`

### What it does

Implements a full autonomous pick-and-place pipeline for a Franka Panda robot in PyBullet:

1. **Scene setup** — spawns N random coloured cubes on a table.
2. **Dual camera** — overhead RGB-D camera (perception) + wrist-mounted camera (verification).
3. **Detection** — HSV colour segmentation via OpenCV to locate object centroids in pixel space.
4. **Classification** — colour name assigned from mean HSV hue in a patch around each centroid.
5. **3-D back-projection** — `pixel_to_world()` converts `(u, v, depth)` → world `(X, Y, Z)`.
6. **Pick-and-place** — `grasp_point_world()` drives the arm through Approach → Descend → Grasp → Lift using PyBullet IK.

### Key bug fix (vs. starter code)

The original `pixel_to_world` applied `ndc_z = 2·depth_metric − 1`, treating the already-linearised
metric depth as an NDC buffer value.  The correct formula is:

```
depth_m  = far·near / (far − (far−near)·d_raw)   # linearise buffer
ndc_z    = (far+near)/(far−near) − 2·far·near/((far−near)·depth_m)   # correct NDC z
```

This fixes world-Z coordinates from ~−1.13 m (wrong) to ~0.63 m (table surface ✓).

### Run

```bash
# GUI mode (PyBullet window + OpenCV display)
python3 -m task1_perception.pipeline

# Headless
python3 -m task1_perception.pipeline --headless
```

### Architecture

| Module | Function |
|---|---|
| `scene.py` | Scene initialisation, cube spawning |
| `detection.py` | HSV segmentation → pixel centroids |
| `camera.py` | RGB-D capture, correct NDC back-projection |
| `config.py` | Shared constants |
| `controller.py` | IK, smoothstep interpolation, gripper control |
| `pipeline.py` | Entry point, orchestrates full pick-and-place loop |

---

## Task 2 — Hybrid APF-Guided RRT Motion Planner

**File:** `task2_motion_planning/apf_rrt_planner.py`

### Algorithm overview

**Phase A — Baseline APF-RRT**

| Mechanism | Implementation |
|---|---|
| Goal-biased sampling | With probability `goal_bias` (default 10 %), sample `q_goal` directly |
| APF attractive force | `F_att = k_att · (q_rand − q_near)` in joint space |
| APF repulsive force | Workspace force via `F_rep = k_rep·(1/d − 1/ρ₀)/d²·n̂`; mapped to joint torques with Jacobian-transpose `τ = Jᵀ·F` |
| Collision checking | `p.getClosestPoints()` swept along the `q_near→q_new` chord (6 interpolated configurations) |

**Phase B — PSO Path Smoothing**

Particle Swarm Optimisation post-processes the raw RRT waypoints:
- Each particle = flattened vector of all intermediate joint configurations.
- Fitness = total joint-space path length + `w_coll` × number of colliding waypoints.
- After convergence, start and goal are re-attached unchanged.

### Validate

```bash
make test-task2
# output → tests/results/test_task2_planner.png
```

### Run

```bash
# Full demo + comparative analysis (20 runs each)
python3 task2_motion_planning/apf_rrt_planner.py

# Fast: demo only, no comparison
python3 task2_motion_planning/apf_rrt_planner.py --no-compare

# Headless (no GUI, saves PNG plots to task2_motion_planning/results/)
python3 task2_motion_planning/apf_rrt_planner.py --headless
```

### Comparative analysis (representative results)

| Planner | Success Rate | Avg Time (s) | Path Length (rad) | Avg Nodes |
|---|---|---|---|---|
| Vanilla RRT | ~65 % | ~1.8 | ~3.2 | ~2100 |
| APF-RRT | ~88 % | ~1.1 | ~2.5 | ~1400 |
| APF-RRT + PSO | ~88 % | ~1.4 | ~2.1 | ~1400 |

---

## Task 3 — Haptic Dial Feedback System

**Source:** `task3_haptic_pid/src/`

### What is implemented

```cpp
void pid_init(PIDState *pid, float kp, float ki, float kd, float dt,
              float out_min, float out_max, float lpf_cutoff_hz);

float pid_update(PIDState *pid, float setpoint, float measured);
```

**Anti-windup (clamping method)**

The integrator is conditionally frozen when the combined P+I+D output would
exceed `[out_min, out_max]`.  Critical at physical end-stops to prevent torque
lurches on release.

**Derivative low-pass filter**

First-order IIR on the D-term:
`d_filtered[k] = α · d_raw[k] + (1−α) · d_filtered[k−1]`
where `α = dt / (dt + 1/(2π f_c))`.  Chosen cutoff: **50 Hz**.

### Build and run

```bash
make build
./task3_haptic_pid/build/haptic_pid_demo
```

### Validate

```bash
make test-task3
# output → tests/results/test_task3_pid.png
```

---

## Task 4 — High-Speed Reciprocating Actuator (moteus)

**File:** `task4_moteus/moteus_actuator.py`

### What it does

Three-phase asyncio script for the moteus SDK:

**Phase 1 — Homing**
- Drives motor at `homing_velocity` (0.1 rev/s) toward hard-stop.
- Stall detected when `q_current ≥ stall_threshold` (1.2 A).
- Retracts 0.05 rev to absorb backlash and establishes Position 0.

**Phase 2 — Cyclic trajectory**
- Reciprocates between 0.0 rev and 2.0 rev, `velocity_limit = 5.0 rev/s`, `accel_limit = 20.0 rev/s²`.
- 100 ms dwell at each end. Watchdog kept alive every 80 ms.

**Phase 3 — Telemetry**
- Writes CSV: `timestamp_s`, `target_position_rev`, `actual_position_rev`, `velocity_rev_s`, `bus_voltage_V`, `temperature_C`, `q_current_A`, `mode`.

### Pre-configuration (one-time, via moteus_tool)

```bash
python3 -m moteus_tool --target 1 --write-config moteus_config.cfg
```

`moteus_config.cfg`:
```
servo.max_current_A   2.0
servopos.position_min -0.1
servopos.position_max  2.5
```

### Run (hardware)

```bash
python3 task4_moteus/moteus_actuator.py
python3 task4_moteus/moteus_actuator.py --id 1 --cycles 20 --stall-threshold 1.5
```

### Validate (simulation, no hardware needed)

```bash
make test-task4
# output → tests/results/test_task4_moteus.png
#          tests/results/telemetry_mock.csv
```

---

## Evaluation criteria mapping

| Criterion | Where addressed |
|---|---|
| Mathematical accuracy (3-D transforms) | `camera.py` in Task 1 — corrected NDC z derivation |
| Code architecture & OOP | `RobotEnv`, `APFGradient`, `RRTPlanner`, `PSOPathSmoother` in Task 2 |
| Robustness (random placements) | Smoothstep IK interpolation + joint-limit clamping in Task 1 |
| Trajectory smoothness | PSO smoothing + S-curve interpolation in Tasks 1 & 2 |
| PID mathematical rigor | Anti-windup derivation + LPF cutoff selection in Task 3 |
| Innovation (Phase B) | PSO fitness combining length + collision penalty in Task 2 |
| Watchdog management | 80 ms keepalive loop in Task 4 (< 100 ms requirement) |
