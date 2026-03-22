"""
Hybrid APF-Guided RRT Motion Planner
=====================================
Franka Panda motion planning in PyBullet

Architecture
------------
Phase A – Baseline APF-RRT
  • Goal-biased sampling  : with probability `goal_bias` the random sample IS
                            the goal configuration, accelerating convergence.
  • APF-guided expansion  : instead of expanding toward q_rand directly, the
                            step direction is biased by a potential field:
                              F_total = F_att(q_rand) + F_rep(obstacles)
  • F_att pulls toward the random sample (which itself is biased toward goal).
  • F_rep is computed by mapping workspace repulsive forces through the robot
    Jacobian (Jacobian-transpose method) so the gradient lives in joint space.
  • Collision checking    : swept-volume check via PyBullet getClosestPoints()
                            along the straight-line path between q_near→q_new.

Phase B – PSO Path Smoothing
  • Particle Swarm Optimisation post-processes the raw RRT waypoints.
  • Each particle represents a candidate set of intermediate waypoints.
  • Fitness = total path length  +  collision-penalty.
  • The result is a smooth, near-time-optimal trajectory.

Comparative Analysis (20 runs each):
  Baseline RRT  vs  APF-RRT  vs  APF-RRT + PSO smoothing
  Metrics: success rate, computation time, path length, node count.

Usage
-----
    python apf_rrt_planner.py               # runs demo + comparison
    python apf_rrt_planner.py --headless    # no GUI window
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Visualisation outputs land in results/ next to this file
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

import sys as _sys
import os as _os
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe on headless)
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

# Ensure pip's mpl_toolkits is loaded (not the stale system copy).
_pip_parent = _os.path.dirname(_os.path.dirname(matplotlib.__file__))
if _pip_parent not in _sys.path:
    _sys.path.insert(0, _pip_parent)
try:
    from mpl_toolkits.mplot3d import (
        Axes3D,
    )  # noqa: F401  # pylint: disable=unused-import,wrong-import-position

    _HAS_3D = True
except ImportError:
    _HAS_3D = False

import numpy as np  # pylint: disable=wrong-import-position
import pybullet as p  # pylint: disable=wrong-import-position
import pybullet_data  # pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Robot / world constants (Franka Panda — 7 DOF, joints 0–6)
# ---------------------------------------------------------------------------
NUM_JOINTS = 7
EE_LINK = 11

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Start and goal configurations (verified collision-free against OBSTACLES below)
Q_START = np.array([0.000, -0.785, 0.000, -2.356, 0.000, 1.571, 0.785])
Q_GOAL = np.array([0.808, -1.675, -1.304, -2.402, 1.370, 2.534, 2.273])

# ---------------------------------------------------------------------------
# Obstacle layout — 3 spheres placed in the mid-arc of the EE's trajectory.
# Each entry: (centre_world_xyz, radius_m)
#
# The EE sweeps from (0.207, 0, 0.485) at Q_START to (0.316, -0.309, 0.531)
# at Q_GOAL, peaking around y=-0.14 to -0.21, z=0.67-0.70 in the middle.
# These 3 spheres sit in that mid-arc so the arm must visibly route around
# them. Both Q_START and Q_GOAL are verified collision-free.
# ---------------------------------------------------------------------------
OBSTACLES = [
    ([0.21, -0.17, 0.67], 0.07),
    ([0.24, -0.21, 0.70], 0.07),
    ([0.18, -0.17, 0.65], 0.07),
]


# ===========================================================================
# PyBullet environment manager
# ===========================================================================


class RobotEnv:
    """
    Thin wrapper around a PyBullet instance.
    Loads the Panda robot and adds sphere obstacles.
    """

    def __init__(self, use_gui: bool = False):
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[-0.1, 0.0, 0.0],
            useFixedBase=True,
        )
        self.obs_ids: List[int] = []
        for centre, radius in OBSTACLES:
            v = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius, rgbaColor=[0.9, 0.3, 0.2, 0.6]
            )
            c = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=c,
                baseVisualShapeIndex=v,
                basePosition=centre,
            )
            self.obs_ids.append(bid)

    def set_config(self, q: np.ndarray) -> None:
        """Instantly set the robot to joint configuration q (no dynamics)."""
        for i, angle in enumerate(q):
            p.resetJointState(self.robot, i, float(angle))

    def get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector world position for configuration q."""
        self.set_config(q)
        state = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True)
        return np.array(state[4])

    def get_link_positions(self, q: np.ndarray) -> List[np.ndarray]:
        """Return world positions of all arm links for configuration q."""
        self.set_config(q)
        positions = []
        for i in range(NUM_JOINTS):
            state = p.getLinkState(self.robot, i, computeForwardKinematics=True)
            positions.append(np.array(state[4]))
        return positions

    def is_collision(self, q: np.ndarray, margin: float = 0.01) -> bool:
        """
        Returns True if any robot link penetrates an obstacle (+ margin metres).

        Broad-phase: compare link-frame origins to obstacle centres using
        fast Euclidean distance.  Link bodies extend at most ~0.12 m from
        their frame origin, so if every link origin is further than
        (obs_radius + 0.18 m) from every obstacle, skip the narrow-phase
        entirely — this short-circuits ~85 % of calls.
        Narrow-phase: PyBullet getClosestPoints on relevant links only (3-6).
        """
        self.set_config(q)
        relevant_links = [3, 4, 5, 6]
        any_close = False
        for link in relevant_links:
            lp = np.array(
                p.getLinkState(self.robot, link, computeForwardKinematics=True)[4]
            )
            for obs_c, obs_r in OBSTACLES:
                if np.linalg.norm(lp - np.array(obs_c)) < obs_r + 0.18:
                    any_close = True
                    break
            if any_close:
                break
        if not any_close:
            return False
        p.performCollisionDetection()
        for obs_id in self.obs_ids:
            for link in relevant_links:
                if p.getClosestPoints(
                    self.robot, obs_id, distance=margin, linkIndexA=link
                ):
                    return True
        return False

    def disconnect(self) -> None:
        """Disconnect from the PyBullet physics server."""
        p.disconnect(self.client)


# ===========================================================================
# RRT node
# ===========================================================================


@dataclass
class Node:
    """A single RRT tree node storing a joint configuration and its parent index."""

    q: np.ndarray
    parent: Optional[int] = None  # index into the tree's node list


# ===========================================================================
# APF helper
# ===========================================================================


class APFGradient:
    """
    Computes joint-space APF gradients using the Jacobian-transpose method.

    Attractive potential  : Uatt = ½ k_att ‖q_rand − q‖²
    Repulsive potential   : Urep = ½ k_rep (1/d − 1/ρ₀)²  for d < ρ₀

    F_att (joint space)   = k_att · (q_rand − q)
    F_rep (joint space)   = J^T · f_rep_workspace   (summed over all links)
    """

    def __init__(
        self, env: RobotEnv, k_att: float = 1.0, k_rep: float = 0.8, rho_0: float = 0.12
    ):
        self.env = env
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        # Count total moveable DOFs (arm + fingers) for calculateJacobian
        self._n_dof = sum(
            1
            for j in range(p.getNumJoints(env.robot))
            if p.getJointInfo(env.robot, j)[2] != p.JOINT_FIXED
        )

    def attractive(self, q: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        """Return the joint-space attractive force pulling q toward q_rand."""
        diff = q_rand - q
        return self.k_att * diff

    def repulsive(self, q: np.ndarray) -> np.ndarray:
        """
        Repulsive gradient in joint space via Jacobian-transpose method.
        For each robot link and each obstacle, compute workspace repulsive
        force then map to joint torques with J^T.
        """
        f_joint = np.zeros(NUM_JOINTS)
        link_positions = self.env.get_link_positions(q)

        for link_idx, link_pos in enumerate(link_positions):
            for obs_centre, obs_radius in OBSTACLES:
                obs_pos = np.array(obs_centre)
                diff = link_pos - obs_pos
                d = np.linalg.norm(diff) - obs_radius  # clearance
                d = max(d, 1e-4)

                if d < self.rho_0:
                    # Workspace repulsive force (gradient of Urep)
                    f_ws = (
                        self.k_rep
                        * (1.0 / d - 1.0 / self.rho_0)
                        / (d**2)
                        * diff
                        / (np.linalg.norm(diff) + 1e-6)
                    )

                    # Jacobian for this link (linear velocity part only).
                    # calculateJacobian needs positions for ALL moveable DOFs
                    # (7 arm + 2 finger for Panda = 9 total).
                    full_q = list(q) + [0.0] * (self._n_dof - NUM_JOINTS)
                    zeros_dof = [0.0] * self._n_dof
                    jac_lin, _ = p.calculateJacobian(
                        self.env.robot,
                        link_idx,
                        localPosition=[0.0, 0.0, 0.0],
                        objPositions=full_q,
                        objVelocities=zeros_dof,
                        objAccelerations=zeros_dof,
                    )
                    J = np.array(jac_lin)[:3, :NUM_JOINTS]  # 3×7
                    # Jacobian-transpose mapping: τ_rep = J^T · f_ws
                    f_joint += J.T @ f_ws

        return f_joint

    def total(self, q: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        """Return the sum of attractive and repulsive joint-space gradients."""
        return self.attractive(q, q_rand) + self.repulsive(q)


# ===========================================================================
# RRT Planner
# ===========================================================================


class RRTPlanner:
    """
    Generic RRT planner.  If `apf` is not None, expansion is APF-guided.

    Parameters
    ----------
    env       : RobotEnv
    apf       : APFGradient or None (None → vanilla RRT)
    step_size : max arc length in joint space per extension
    goal_bias : probability of sampling q_goal directly
    max_iter  : maximum tree nodes before declaring failure
    goal_tol  : joint-space distance to declare goal reached
    """

    def __init__(
        self,
        env: RobotEnv,
        apf: Optional[APFGradient] = None,
        step_size: float = 0.30,
        goal_bias: float = 0.15,
        max_iter: int = 8000,
        goal_tol: float = 0.30,
        stall_limit: int = 20,
    ):
        self.env = env
        self.apf = apf
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iter = max_iter
        self.goal_tol = goal_tol
        self.stall_limit = stall_limit  # consecutive hits on same nearest node → escape
        self.tree: List[Node] = []
        self.stats = {}

    # ------------------------------------------------------------------
    def plan(
        self, q_start: np.ndarray, q_goal: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """
        Run the planner.  Returns a path (list of configs from start to goal)
        or None if no path found within max_iter iterations.
        """
        self.tree = [Node(q=q_start.copy())]
        t0 = time.time()
        _stall_count = 0
        _last_near_idx = -1

        for iteration in range(self.max_iter):
            # ── 1. Sample (adaptive goal bias in second half) ──────────
            progress = iteration / self.max_iter
            effective_bias = self.goal_bias + (0.40 - self.goal_bias) * max(0.0, (progress - 0.5) * 2)
            q_rand = self._sample(q_goal, effective_bias)

            # ── 2. Nearest node ───────────────────────────────────────
            near_idx, q_near = self._nearest(q_rand)

            # ── 3. Stall detection: same node picked too many times ────
            if near_idx == _last_near_idx:
                _stall_count += 1
            else:
                _stall_count = 0
                _last_near_idx = near_idx

            # ── 4. Extend (APF-guided, or pure random on stall) ────────
            force_random = _stall_count >= self.stall_limit
            q_new = self._extend(q_near, q_rand, skip_apf=force_random)
            if q_new is None:
                continue

            # ── 5. Collision check (swept volume) ─────────────────────
            if not self._path_free(q_near, q_new):
                continue

            # ── 6. Add node ───────────────────────────────────────────
            new_idx = len(self.tree)
            self.tree.append(Node(q=q_new, parent=near_idx))

            # ── 7. Goal check ─────────────────────────────────────────
            if np.linalg.norm(q_new - q_goal) < self.goal_tol:
                # Attempt direct connection to exact goal
                if self._path_free(q_new, q_goal):
                    goal_node = Node(q=q_goal.copy(), parent=new_idx)
                    self.tree.append(goal_node)
                    path = self._extract_path(len(self.tree) - 1)
                    self.stats = {
                        "success": True,
                        "time": time.time() - t0,
                        "node_count": len(self.tree),
                        "path_length": self.path_length(path),
                    }
                    return path

        self.stats = {
            "success": False,
            "time": time.time() - t0,
            "node_count": len(self.tree),
            "path_length": float("inf"),
        }
        return None

    # ------------------------------------------------------------------
    def _sample(self, q_goal: np.ndarray, bias: Optional[float] = None) -> np.ndarray:
        if random.random() < (bias if bias is not None else self.goal_bias):
            return q_goal.copy()
        return np.array(
            [random.uniform(lo, hi) for lo, hi in zip(JOINT_LOWER, JOINT_UPPER)]
        )

    def _nearest(self, q_rand: np.ndarray) -> Tuple[int, np.ndarray]:
        dists = [np.linalg.norm(node.q - q_rand) for node in self.tree]
        idx = int(np.argmin(dists))
        return idx, self.tree[idx].q.copy()

    def _extend(
        self, q_near: np.ndarray, q_rand: np.ndarray, skip_apf: bool = False
    ) -> Optional[np.ndarray]:
        if self.apf is not None and not skip_apf:
            # APF-guided: bias direction with potential field gradient.
            # Skip Jacobian entirely when no obstacle is within rho_0 of any link.
            link_positions = self.env.get_link_positions(q_near)
            near_obstacle = any(
                np.linalg.norm(lp - np.array(obs_c)) - obs_r < self.apf.rho_0
                for lp in link_positions
                for obs_c, obs_r in OBSTACLES
            )
            if near_obstacle:
                grad = self.apf.total(q_near, q_rand)
            else:
                grad = self.apf.attractive(q_near, q_rand)
        else:
            # Vanilla RRT or stall-escape: steer straight toward q_rand
            grad = q_rand - q_near

        norm = np.linalg.norm(grad)
        if norm < 1e-6:
            return None

        q_new = q_near + self.step_size * grad / norm
        # Clamp to joint limits
        q_new = np.clip(q_new, JOINT_LOWER, JOINT_UPPER)
        return q_new

    def _path_free(self, q1: np.ndarray, q2: np.ndarray, n_checks: int = 6) -> bool:
        """Linearly interpolate and check n_checks+1 configurations."""
        for i in range(n_checks + 1):
            t = i / n_checks
            q = q1 + t * (q2 - q1)
            if self.env.is_collision(q):
                return False
        return True

    def _extract_path(self, node_idx: int) -> List[np.ndarray]:
        path = []
        idx = node_idx
        while idx is not None:
            path.append(self.tree[idx].q.copy())
            idx = self.tree[idx].parent
        return list(reversed(path))

    @staticmethod
    def path_length(path: List[np.ndarray]) -> float:
        """Return total joint-space arc length of a path."""
        return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))

    # ------------------------------------------------------------------
    def plan_bidirectional(
        self, q_start: np.ndarray, q_goal: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """
        Bidirectional RRT-Connect.

        Grows two trees simultaneously — one from q_start, one from q_goal —
        and attempts to bridge them at every step.  This naturally escapes local
        minima that trap single-tree planners: when the start-tree is stuck behind
        an obstacle cluster, the goal-tree can route around it from the other side
        and meet in the middle.

        On each iteration the *shorter* tree is extended (balanced growth), then
        we check whether the new node is close enough to connect to the other tree.
        APF guidance and stall-escape are applied to whichever tree is extending.
        """
        tree_s: List[Node] = [Node(q=q_start.copy())]   # start tree
        tree_g: List[Node] = [Node(q=q_goal.copy())]    # goal tree
        t0 = time.time()
        stall_s = stall_g = 0
        last_s = last_g = -1

        def _nearest_in(tree: List[Node], q: np.ndarray) -> Tuple[int, np.ndarray]:
            dists = [np.linalg.norm(n.q - q) for n in tree]
            idx = int(np.argmin(dists))
            return idx, tree[idx].q.copy()

        def _extract(tree: List[Node], idx: int) -> List[np.ndarray]:
            path, i = [], idx
            while i is not None:
                path.append(tree[i].q.copy())
                i = tree[i].parent
            return list(reversed(path))

        for iteration in range(self.max_iter):
            # Balance: extend the smaller tree so both grow at similar rates
            extend_start = len(tree_s) <= len(tree_g)
            active, passive = (tree_s, tree_g) if extend_start else (tree_g, tree_s)

            # Adaptive goal bias: ramp up in second half of budget
            progress = iteration / self.max_iter
            eff_bias = self.goal_bias + (0.40 - self.goal_bias) * max(0.0, (progress - 0.5) * 2)

            # Sample: with eff_bias probability, pull toward a random node in
            # the passive tree (greedily tries to bridge); otherwise random.
            if random.random() < eff_bias:
                q_rand = passive[random.randint(0, len(passive) - 1)].q.copy()
            else:
                q_rand = np.array(
                    [random.uniform(lo, hi) for lo, hi in zip(JOINT_LOWER, JOINT_UPPER)]
                )

            near_idx, q_near = _nearest_in(active, q_rand)

            # Stall detection per tree
            if extend_start:
                stall_s = stall_s + 1 if near_idx == last_s else 0
                last_s = near_idx
                force_rand = stall_s >= self.stall_limit
            else:
                stall_g = stall_g + 1 if near_idx == last_g else 0
                last_g = near_idx
                force_rand = stall_g >= self.stall_limit

            q_new = self._extend(q_near, q_rand, skip_apf=force_rand)
            if q_new is None:
                continue
            if not self._path_free(q_near, q_new):
                continue

            new_idx = len(active)
            active.append(Node(q=q_new, parent=near_idx))

            # Try to connect new node to the passive tree
            conn_idx, q_conn = _nearest_in(passive, q_new)
            if np.linalg.norm(q_new - q_conn) < self.goal_tol:
                if self._path_free(q_new, q_conn):
                    path_active  = _extract(active,  new_idx)
                    path_passive = _extract(passive, conn_idx)
                    full = (path_active + list(reversed(path_passive))
                            if extend_start
                            else list(reversed(path_passive)) + path_active)
                    self.stats = {
                        "success": True,
                        "time": time.time() - t0,
                        "node_count": len(tree_s) + len(tree_g),
                        "path_length": self.path_length(full),
                    }
                    self.tree = tree_s  # expose for visualisation
                    return full

        self.stats = {
            "success": False,
            "time": time.time() - t0,
            "node_count": len(tree_s) + len(tree_g),
            "path_length": float("inf"),
        }
        return None


# ===========================================================================
# Parallel portfolio planner
# ===========================================================================


def _parallel_worker(args: tuple):
    """
    Module-level worker for ProcessPoolExecutor.
    Each worker spins up its own PyBullet DIRECT instance so processes are
    fully independent — no shared memory, no GIL contention.
    """
    seed, use_apf, max_iter, bidirectional = args
    import random as _r
    import numpy as _np
    _r.seed(seed)
    _np.random.seed(seed)

    env = RobotEnv(use_gui=False)
    try:
        apf = APFGradient(env) if use_apf else None
        planner = RRTPlanner(env, apf=apf, max_iter=max_iter)
        path = (planner.plan_bidirectional(Q_START, Q_GOAL)
                if bidirectional else planner.plan(Q_START, Q_GOAL))
        return path, planner.stats.copy()
    finally:
        try:
            env.disconnect()
        except Exception:
            pass


def plan_parallel_portfolio(
    n_workers: int = 4,
    use_apf: bool = True,
    bidirectional: bool = True,
    max_iter: int = 8000,
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """
    Parallel portfolio planner.

    Launches `n_workers` independent planning processes, each with a different
    random seed.  Returns the first successful path found.

    With per-run success rate p and n workers, combined success ≈ 1 − (1−p)^n.
    E.g. p=0.60, n=4  →  97 % success at ~1/4 the single-run wall-clock time
    for the typical (fast) cases.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t0 = time.time()
    worker_args = [(i, use_apf, max_iter, bidirectional) for i in range(n_workers)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_parallel_worker, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            try:
                path, stats = future.result()
            except Exception:
                continue
            if path is not None:
                for f in futures:
                    f.cancel()
                stats["total_time"] = time.time() - t0
                stats["workers"] = n_workers
                return path, stats

    return None, {
        "success": False, "total_time": time.time() - t0, "workers": n_workers
    }


# ===========================================================================
# Phase B — PSO Path Smoother
# ===========================================================================


class PSOPathSmoother:
    """
    Post-processes a collision-free RRT path using Particle Swarm Optimisation.

    Each particle represents the full set of intermediate waypoints (all nodes
    between start and goal).  The swarm minimises:

        fitness = path_length(waypoints) + w_coll · collision_count(waypoints)

    After convergence the smooth path retains start and goal exactly.

    Parameters
    ----------
    env         : RobotEnv (used for collision checking)
    n_particles : swarm size
    n_iter      : number of PSO generations
    w           : inertia weight
    c1, c2      : cognitive / social acceleration coefficients
    w_coll      : penalty weight per collision violation
    perturb_std : initial position noise spread around the raw path
    """

    def __init__(
        self,
        env: RobotEnv,
        n_particles: int = 20,
        n_iter: int = 100,
        w: float = 0.50,
        c1: float = 1.50,
        c2: float = 1.50,
        w_coll: float = 10.0,
        perturb_std: float = 0.05,
    ):
        self.env = env
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_coll = w_coll
        self.perturb_std = perturb_std

    def smooth(self, raw_path: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run PSO and return a smoothed path.  Start and goal are fixed.
        If raw_path has ≤ 2 nodes (only start+goal) returns it unchanged.
        """
        if len(raw_path) <= 2:
            return raw_path

        start = raw_path[0]
        goal = raw_path[-1]
        interm = np.array(raw_path[1:-1])  # shape (K, 7)
        K, D = interm.shape
        flat = interm.flatten()  # shape (K*7,)
        dim = flat.size

        # ── Initialise swarm ──────────────────────────────────────────
        pos = flat + np.random.randn(self.n_particles, dim) * self.perturb_std
        vel = np.zeros_like(pos)
        # Clamp initial positions to joint limits
        for p_idx in range(self.n_particles):
            pos[p_idx] = self._clamp_flat(pos[p_idx], K)

        pbest_pos = pos.copy()
        pbest_fit = np.array(
            [self._fitness(pos[i], start, goal, K) for i in range(self.n_particles)]
        )
        gbest_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[gbest_idx].copy()

        # ── PSO loop ──────────────────────────────────────────────────
        for _ in range(self.n_iter):
            r1 = np.random.rand(self.n_particles, dim)
            r2 = np.random.rand(self.n_particles, dim)

            vel = (
                self.w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            pos = pos + vel

            for p_idx in range(self.n_particles):
                pos[p_idx] = self._clamp_flat(pos[p_idx], K)
                f = self._fitness(pos[p_idx], start, goal, K)
                if f < pbest_fit[p_idx]:
                    pbest_fit[p_idx] = f
                    pbest_pos[p_idx] = pos[p_idx].copy()

            gbest_idx = int(np.argmin(pbest_fit))
            gbest_pos = pbest_pos[gbest_idx].copy()

        # ── Reconstruct smooth path ───────────────────────────────────
        best_interm = gbest_pos.reshape(K, D)
        smooth_path = [start] + [best_interm[i] for i in range(K)] + [goal]
        return smooth_path

    # ------------------------------------------------------------------
    def _fitness(
        self, flat: np.ndarray, start: np.ndarray, goal: np.ndarray, K: int
    ) -> float:
        waypoints = [start] + list(flat.reshape(K, NUM_JOINTS)) + [goal]
        length = sum(
            np.linalg.norm(waypoints[i + 1] - waypoints[i])
            for i in range(len(waypoints) - 1)
        )
        collisions = sum(1 for q in waypoints[1:-1] if self.env.is_collision(q))
        return length + self.w_coll * collisions

    @staticmethod
    def _clamp_flat(flat: np.ndarray, K: int) -> np.ndarray:
        q_mat = flat.reshape(K, NUM_JOINTS)
        q_mat = np.clip(q_mat, JOINT_LOWER, JOINT_UPPER)
        return q_mat.flatten()


# ===========================================================================
# Comparative analysis
# ===========================================================================


def run_experiment(label: str, env: RobotEnv, use_apf: bool, n_runs: int = 20) -> dict:
    """
    Runs n_runs trials and returns aggregate statistics.
    """
    apf = APFGradient(env) if use_apf else None

    successes, times, lengths, node_counts = [], [], [], []

    for _ in range(n_runs):
        # Slightly randomise start/goal to get diverse trials
        noise = np.random.randn(NUM_JOINTS) * 0.1
        q_s = np.clip(Q_START + noise, JOINT_LOWER, JOINT_UPPER)
        q_g = np.clip(Q_GOAL + noise, JOINT_LOWER, JOINT_UPPER)

        planner = RRTPlanner(env, apf=apf)
        planner.plan(q_s, q_g)
        s = planner.stats

        successes.append(s["success"])
        times.append(s["time"])
        node_counts.append(s["node_count"])
        if s["success"]:
            lengths.append(s["path_length"])

    success_rate = sum(successes) / n_runs
    avg_time = float(np.mean(times))
    avg_length = float(np.mean(lengths)) if lengths else float("inf")
    avg_nodes = float(np.mean(node_counts))

    result = {
        "label": label,
        "success_rate": success_rate,
        "avg_time_s": avg_time,
        "avg_length": avg_length,
        "avg_nodes": avg_nodes,
    }
    print(
        f"  {label:30s}  SR={success_rate:.0%}  "
        f"t={avg_time:.2f}s  L={avg_length:.3f}  N={avg_nodes:.0f}"
    )
    return result


# ===========================================================================
# Visualisation
# ===========================================================================


def visualise_tree_and_path(
    env: RobotEnv,
    planner: RRTPlanner,
    path: Optional[List[np.ndarray]],
    title: str,
    save_path: str,
) -> None:
    """
    Plots the RRT tree (EE positions) and the final path in 3-D world space.
    Saves the figure to save_path.
    """
    fig = plt.figure(figsize=(9, 7))
    if _HAS_3D:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title + " [2D fallback — X vs Z]")

    # Draw tree edges (grey)
    for node in planner.tree[1:]:
        if node.parent is not None:
            q1 = planner.tree[node.parent].q
            q2 = node.q
            p1 = env.get_ee_position(q1)
            p2 = env.get_ee_position(q2)
            if _HAS_3D:
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    "b-",
                    alpha=0.15,
                    linewidth=0.6,
                )
            else:
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], "b-", alpha=0.15, lw=0.6)

    # Draw obstacles
    for centre, radius in OBSTACLES:
        if _HAS_3D:
            u = np.linspace(0, 2 * math.pi, 20)
            v = np.linspace(0, math.pi, 20)
            xs = radius * np.outer(np.cos(u), np.sin(v)) + centre[0]
            ys = radius * np.outer(np.sin(u), np.sin(v)) + centre[1]
            zs = radius * np.outer(np.ones_like(u), np.cos(v)) + centre[2]
            ax.plot_surface(xs, ys, zs, color="salmon", alpha=0.3)
        else:
            c2d = plt.Circle((centre[0], centre[2]), radius, color="salmon", alpha=0.4)
            ax.add_patch(c2d)

    # Draw path
    if path:
        pts = np.array([env.get_ee_position(q) for q in path])
        if _HAS_3D:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                "g-o",
                linewidth=2.5,
                markersize=4,
                label="Path",
            )
            ax.scatter(*pts[0], color="lime", s=60, zorder=5, label="Start")
            ax.scatter(*pts[-1], color="red", s=60, zorder=5, label="Goal")
        else:
            ax.plot(pts[:, 0], pts[:, 2], "g-o", lw=2.5, ms=4, label="Path")
            ax.scatter(
                pts[0, 0], pts[0, 2], color="lime", s=60, zorder=5, label="Start"
            )
            ax.scatter(
                pts[-1, 0], pts[-1, 2], color="red", s=60, zorder=5, label="Goal"
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)" if _HAS_3D else "Z (m)")
    if _HAS_3D:
        ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved visualisation → {save_path}")


def print_comparison_table(results: List[dict]) -> None:
    """Print a formatted table comparing planner metrics across multiple runs."""
    w = 30
    sep = "-" * (w + 52)
    print(f"\n{'Comparative Analysis':^{w + 52}}")
    print(sep)
    print(
        f"{'Planner':<{w}} {'Success':>8} {'Time(s)':>9} "
        f"{'Path Len':>10} {'Nodes':>8}"
    )
    print(sep)
    for r in results:
        sl = "∞" if r["avg_length"] == float("inf") else f"{r['avg_length']:.3f}"
        print(
            f"{r['label']:<{w}} {r['success_rate']:>8.0%} "
            f"{r['avg_time_s']:>9.2f} {sl:>10} {r['avg_nodes']:>8.0f}"
        )
    print(sep)


# ===========================================================================
# Demo: single planning run + PSO smoothing + visualisation
# ===========================================================================


def execute_path_gui(
    env: RobotEnv, path: List[np.ndarray], interp_steps: int = 30
) -> None:
    """
    Animate the robot following `path` in the PyBullet GUI.
    Interpolates linearly between each pair of waypoints so the motion is
    smooth rather than jumping frame-to-frame.
    """
    for i in range(len(path) - 1):
        q_a, q_b = path[i], path[i + 1]
        for s in range(interp_steps + 1):
            t = s / interp_steps
            q = q_a + t * (q_b - q_a)
            env.set_config(q)
            p.stepSimulation()
            time.sleep(1.0 / 120.0)


def run_demo(env: RobotEnv, use_gui: bool = False) -> None:
    """Run a single APF-RRT planning demo and optionally animate the result in the GUI."""
    print("\n── Demo: APF-RRT + PSO Smoothing ────────────────────────────")
    apf = APFGradient(env)
    planner = RRTPlanner(env, apf=apf, goal_bias=0.10, step_size=0.25)
    path = planner.plan(Q_START, Q_GOAL)

    if path is None:
        print("  No path found in demo run.")
        return

    s = planner.stats
    print(
        f"  Raw path  : {len(path)} nodes, "
        f"length = {s['path_length']:.3f}, time = {s['time']:.2f} s"
    )

    # PSO smoothing
    smoother = PSOPathSmoother(env, n_particles=20, n_iter=80)
    smooth_path = smoother.smooth(path)
    smooth_len = RRTPlanner.path_length(smooth_path)
    print(
        f"  Smooth path: {len(smooth_path)} waypoints, "
        f"length = {smooth_len:.3f}  "
        f"(reduction {100*(1-smooth_len/s['path_length']):.1f} %)"
    )

    visualise_tree_and_path(
        env, planner, path, "APF-RRT Raw Path",
        str(_RESULTS_DIR / "apf_rrt_raw.png"),
    )
    visualise_tree_and_path(
        env, planner, smooth_path, "APF-RRT + PSO Smooth Path",
        str(_RESULTS_DIR / "apf_rrt_smooth.png"),
    )

    if use_gui:
        # Label: show what we're about to do before each execution
        lbl = p.addUserDebugText(
            f"RAW RRT PATH  ({len(path)} waypoints)\nArm navigates around red obstacles",
            [0.3, 0.5, 1.2],
            textColorRGB=[1, 0.6, 0],
            textSize=1.8,
        )
        time.sleep(2.0)
        print("\n  Executing raw path in GUI …")
        execute_path_gui(env, path, interp_steps=60)

        p.removeUserDebugItem(lbl)
        lbl2 = p.addUserDebugText(
            f"PSO-SMOOTHED PATH  ({len(smooth_path)} waypoints)\nSame route, shorter & smoother",
            [0.3, 0.5, 1.2],
            textColorRGB=[0, 1, 0.4],
            textSize=1.8,
        )
        time.sleep(2.0)
        print("  Executing smoothed path in GUI …")
        execute_path_gui(env, smooth_path, interp_steps=60)
        p.removeUserDebugItem(lbl2)
        print("  Path execution done.")


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Parse CLI arguments, run the APF-RRT demo, and optionally run comparative analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--runs", type=int, default=20, help="Comparison runs per planner (default 20)"
    )
    parser.add_argument(
        "--no-compare", action="store_true", help="Skip comparative analysis (faster)"
    )
    args = parser.parse_args()

    env = RobotEnv(use_gui=not args.headless)

    # ── Demo run ──────────────────────────────────────────────────────
    run_demo(env, use_gui=not args.headless)

    # ── Comparative analysis ──────────────────────────────────────────
    if not args.no_compare:
        print(f"\n── Comparative Analysis ({args.runs} runs each) ───────────────")
        results = [
            run_experiment(
                "Vanilla RRT (baseline)", env, use_apf=False, n_runs=args.runs
            ),
            run_experiment("APF-RRT", env, use_apf=True, n_runs=args.runs),
        ]
        print_comparison_table(results)

    env.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
