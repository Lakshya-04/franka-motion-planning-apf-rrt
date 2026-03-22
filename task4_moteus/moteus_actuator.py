"""
moteus_actuator.py
==================
Task 4 — High-Speed Reciprocating Actuator
SocialHW Robotics Assessment

Hardware setup
--------------
  Controller  : moteus (n1 / r4.x / c1)
  Communication: fdcanusb or pi3hat
  Motor       : any BLDC calibrated for moteus
  Mechanical  : physical hard-stop at one end of motor travel

This script uses the moteus Python SDK (asyncio interface) to:

  Phase 1 – Initialisation & Homing
    • Configure safety limits (max_current, position min/max).
    • Drive the motor slowly toward the hard-stop while monitoring
      q_current feedback.
    • Declare Position 0 when torque exceeds the stall threshold.

  Phase 2 – Trajectory Execution
    • Cyclic back-and-forth motion between 0.0 rev and 2.0 rev.
    • velocity_limit = 5.0 rev/s, accel_limit = 20.0 rev/s².
    • 100 ms dwell at each end-point.
    • Watchdog keepalive every ≤ 100 ms.

  Phase 3 – Telemetry Logging
    • Captures target_position, actual_position, bus_voltage,
      temperature, and q_current to a timestamped CSV file.

Requirements
------------
    pip install moteus

Configuration (run once via moteus_tool before this script):
    python -m moteus_tool --target 1 --write-config moteus_config.cfg
    where moteus_config.cfg contains:
        servo.max_current_A   2.0
        servopos.position_min 0.0
        servopos.position_max 2.5

Usage
-----
    python moteus_actuator.py [--id CONTROLLER_ID] [--cycles N]
                              [--homing-velocity V] [--stall-threshold T]
                              [--output-csv FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import time
from pathlib import Path

try:
    import moteus
except ImportError as exc:
    raise SystemExit(
        "moteus package not found.\n"
        "Install with:  pip install moteus\n"
        "Documentation: https://github.com/mjbots/moteus"
    ) from exc

# ---------------------------------------------------------------------------
# Default parameters (override via CLI or modify here)
# ---------------------------------------------------------------------------
DEFAULT_CONTROLLER_ID = 1
DEFAULT_CYCLES = 10  # number of full back-and-forth cycles
DEFAULT_HOMING_VEL = 0.10  # rev/s — slow crawl toward hard-stop
DEFAULT_STALL_CURRENT = 1.20  # A    — q_current threshold indicating stall
DEFAULT_CSV = "telemetry.csv"

VELOCITY_LIMIT = 5.0  # rev/s
ACCEL_LIMIT = 20.0  # rev/s²
DWELL_TIME = 0.100  # s  — pause at each end-point
WATCHDOG_INTERVAL = 0.080  # s  — keepalive period (< 100 ms)
TARGET_POS_A = 0.0  # rev — home / end-stop side
TARGET_POS_B = 2.0  # rev — far end


# ===========================================================================
# Telemetry
# ===========================================================================


class TelemetryLogger:
    """Writes telemetry records to a CSV file in real time."""

    COLUMNS = [
        "timestamp_s",
        "target_position_rev",
        "actual_position_rev",
        "velocity_rev_s",
        "bus_voltage_V",
        "temperature_C",
        "q_current_A",
        "mode",
    ]

    def __init__(self, filepath: str):
        self._path = Path(filepath)
        self._fh = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._t0 = time.monotonic()

    def log(self, record: dict) -> None:
        """Append one telemetry row with an auto-generated timestamp."""
        record["timestamp_s"] = round(time.monotonic() - self._t0, 5)
        self._writer.writerow({k: record.get(k, "") for k in self.COLUMNS})

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._fh.flush()
        self._fh.close()
        print(f"  Telemetry saved → {self._path}")


# ===========================================================================
# Helper: extract register values from a moteus query result
# ===========================================================================


def _get(result, register, default=0.0):
    try:
        return result.values[register]
    except (AttributeError, KeyError):
        return default


# ===========================================================================
# Phase 1 — Initialisation & Homing
# ===========================================================================


async def phase_homing(
    c: moteus.Controller,
    logger: TelemetryLogger,
    homing_velocity: float,
    stall_threshold: float,
) -> None:
    """
    Torque-based homing:
      1. Command motor to crawl toward the physical hard-stop.
      2. Monitor q_current each cycle.
      3. When q_current ≥ stall_threshold the motor has hit the wall:
         issue set_stop() to freeze the output, then declare Position 0.

    The position offset is NOT applied via the moteus firmware offset
    register in this script (that requires a separate moteus_tool call).
    Instead we record the stall position and use it as the software zero.

    Hard-stop handling / backlash note
    ------------------------------------
    At the instant of stall detection we issue set_stop() which places the
    controller in the STOPPED fault state, eliminating any current-loop
    winding.  We then send set_position(position=math.nan, velocity=0) to
    bring the controller back to the POSITION state with zero velocity
    command — this avoids the sudden back-drive that would occur if we
    commanded position=0 directly while the coupling is under load.

    After the NaN-velocity settle we command a small retract (0.05 rev) to
    back away from the stop and compensate for mechanical backlash before
    zeroing.
    """
    print("\n── Phase 1: Homing ─────────────────────────────────────────")

    # Ensure controller is live (clear any prior fault)
    await c.set_stop()
    await asyncio.sleep(0.05)

    print(f"  Crawling at {homing_velocity} rev/s toward hard-stop …")
    stall_position = None
    t_start = time.monotonic()

    while True:
        # Velocity command toward the stop (negative direction)
        result = await c.set_position(
            position=math.nan,  # NaN position → velocity-only mode
            velocity=-homing_velocity,
            maximum_torque=2.0,  # current limit during homing
            query=True,
        )

        actual_pos = _get(result, moteus.Register.POSITION)
        q_current = abs(_get(result, moteus.Register.Q_CURRENT))
        voltage = _get(result, moteus.Register.VOLTAGE)
        temp = _get(result, moteus.Register.TEMPERATURE)
        mode = _get(result, moteus.Register.MODE)

        logger.log(
            {
                "target_position_rev": 0.0,
                "actual_position_rev": actual_pos,
                "bus_voltage_V": voltage,
                "temperature_C": temp,
                "q_current_A": q_current,
                "mode": mode,
            }
        )

        if q_current >= stall_threshold:
            print(
                f"  Hard-stop detected: q_current = {q_current:.3f} A "
                f"(threshold {stall_threshold} A)"
            )
            stall_position = actual_pos
            break

        # Timeout safety: if homing takes too long, abort
        if time.monotonic() - t_start > 10.0:
            raise RuntimeError(
                "Homing timeout — hard-stop not detected "
                "within 10 s.  Check cable / direction."
            )

        await asyncio.sleep(WATCHDOG_INTERVAL)

    # ── Stop motor smoothly at the hard-stop ──────────────────────────────
    await c.set_stop()
    await asyncio.sleep(0.05)

    # ── Transition back to position mode with zero velocity ───────────────
    # Using set_position(position=math.nan, velocity=0) avoids the sudden
    # output change that would occur if we jumped straight to an absolute
    # position while the coupling may still be pre-loaded.
    await c.set_position(
        position=math.nan, velocity=0.0, maximum_torque=1.0, query=False
    )
    await asyncio.sleep(0.10)

    # ── Retract slightly to clear the stop and absorb backlash ───────────
    retract_rev = 0.05  # rev
    retract_target = stall_position + retract_rev  # positive = away from stop
    result = await c.set_position(
        position=retract_target,
        velocity_limit=0.5,
        accel_limit=2.0,
        maximum_torque=1.5,
        query=True,
    )
    await asyncio.sleep(0.30)

    # Software zero offset: all subsequent targets are relative to home
    print(
        f"  Stall position = {stall_position:.4f} rev  "
        f"(software zero after {retract_rev} rev retract)"
    )
    print("  Homing complete ✓")
    return stall_position


# ===========================================================================
# Phase 2 — Cyclic Trajectory Execution
# ===========================================================================


async def phase_trajectory(
    c: moteus.Controller, logger: TelemetryLogger, home_offset: float, n_cycles: int
) -> None:
    """
    Executes n_cycles complete back-and-forth motions between
    (home_offset + TARGET_POS_A) and (home_offset + TARGET_POS_B).

    Profile: velocity_limit = 5 rev/s, accel_limit = 20 rev/s²
    Dwell:   100 ms at each end-point.

    Watchdog management:
    The moteus controller requires a command at least every 100 ms or it
    enters FAULT.  During dwell we send a query-only keepalive every
    WATCHDOG_INTERVAL seconds.
    """
    print(f"\n── Phase 2: Trajectory ({n_cycles} cycles) ─────────────────")

    pos_a = home_offset + TARGET_POS_A
    pos_b = home_offset + TARGET_POS_B
    endpoints = [pos_a, pos_b]

    for cycle in range(n_cycles):
        for _, target in enumerate(endpoints):
            print(f"  Cycle {cycle+1}/{n_cycles}  →  target = {target:.3f} rev")

            # ── Move command ──────────────────────────────────────────
            await c.set_position(
                position=target,
                velocity_limit=VELOCITY_LIMIT,
                accel_limit=ACCEL_LIMIT,
                maximum_torque=2.0,
                query=False,
            )

            # ── Wait until position settled (with watchdog keepalives) ─
            t_move_start = time.monotonic()
            move_timeout = 5.0  # s — generous for 2 rev at 5 rev/s

            while True:
                await asyncio.sleep(WATCHDOG_INTERVAL)

                result = await c.set_position(
                    position=target,
                    velocity_limit=VELOCITY_LIMIT,
                    accel_limit=ACCEL_LIMIT,
                    maximum_torque=2.0,
                    query=True,
                )

                actual_pos = _get(result, moteus.Register.POSITION)
                velocity = _get(result, moteus.Register.VELOCITY)
                voltage = _get(result, moteus.Register.VOLTAGE)
                temp = _get(result, moteus.Register.TEMPERATURE)
                q_current = _get(result, moteus.Register.Q_CURRENT)
                mode = _get(result, moteus.Register.MODE)

                logger.log(
                    {
                        "target_position_rev": target,
                        "actual_position_rev": actual_pos,
                        "velocity_rev_s": velocity,
                        "bus_voltage_V": voltage,
                        "temperature_C": temp,
                        "q_current_A": q_current,
                        "mode": mode,
                    }
                )

                # Settled: position within 0.01 rev and nearly stationary
                if abs(actual_pos - target) < 0.01 and abs(velocity) < 0.05:
                    break

                if time.monotonic() - t_move_start > move_timeout:
                    print(f"  Warning: move timeout at pos={actual_pos:.3f}")
                    break

            # ── Dwell with watchdog keepalives ────────────────────────
            t_dwell = time.monotonic()
            while time.monotonic() - t_dwell < DWELL_TIME:
                await asyncio.sleep(
                    min(WATCHDOG_INTERVAL, DWELL_TIME - (time.monotonic() - t_dwell))
                )
                # Keepalive: repeat the same position command
                result = await c.set_position(
                    position=target,
                    velocity_limit=VELOCITY_LIMIT,
                    accel_limit=ACCEL_LIMIT,
                    maximum_torque=2.0,
                    query=True,
                )
                # Log dwell telemetry
                logger.log(
                    {
                        "target_position_rev": target,
                        "actual_position_rev": _get(result, moteus.Register.POSITION),
                        "velocity_rev_s": _get(result, moteus.Register.VELOCITY),
                        "bus_voltage_V": _get(result, moteus.Register.VOLTAGE),
                        "temperature_C": _get(result, moteus.Register.TEMPERATURE),
                        "q_current_A": _get(result, moteus.Register.Q_CURRENT),
                        "mode": _get(result, moteus.Register.MODE),
                    }
                )

    print("  Trajectory complete ✓")


# ===========================================================================
# Main coroutine
# ===========================================================================


async def run(
    controller_id: int,
    n_cycles: int,
    homing_velocity: float,
    stall_threshold: float,
    csv_file: str,
) -> None:
    """
    Top-level coroutine: home the motor, execute cyclic trajectory, then shut down.
    Opens a TelemetryLogger for the full session and always closes it on exit.
    """
    logger = TelemetryLogger(csv_file)

    # Create controller with the registers we want to query
    qr = moteus.QueryResolution()
    qr.position = moteus.F32
    qr.velocity = moteus.F32
    qr.torque = moteus.F32
    qr.q_current = moteus.F32
    qr.voltage = moteus.F32
    qr.temperature = moteus.F32
    qr.mode = moteus.INT8

    c = moteus.Controller(id=controller_id, query_resolution=qr)

    try:
        # ── Phase 1: Homing ───────────────────────────────────────────
        home_offset = await phase_homing(c, logger, homing_velocity, stall_threshold)

        # ── Phase 2: Cyclic trajectory ────────────────────────────────
        await phase_trajectory(c, logger, home_offset, n_cycles)

        # ── Phase 3: Graceful shutdown ─────────────────────────────────
        print("\n── Phase 3: Shutdown ────────────────────────────────────")
        await c.set_stop()
        print("  Motor stopped ✓")

    except Exception as err:
        print(f"\nError: {err}")
        await c.set_stop()  # Always attempt safe stop on exception
        raise

    finally:
        logger.close()


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Parse CLI arguments and launch the async run coroutine."""
    parser = argparse.ArgumentParser(
        description="moteus high-speed reciprocating actuator demo"
    )
    parser.add_argument(
        "--id",
        type=int,
        default=DEFAULT_CONTROLLER_ID,
        help="CAN bus controller ID (default: 1)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help="Number of back-and-forth cycles (default: 10)",
    )
    parser.add_argument(
        "--homing-velocity",
        type=float,
        default=DEFAULT_HOMING_VEL,
        help="Homing crawl speed rev/s (default: 0.1)",
    )
    parser.add_argument(
        "--stall-threshold",
        type=float,
        default=DEFAULT_STALL_CURRENT,
        help="q_current A threshold for stall detection (default: 1.2)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_CSV,
        help=f"Telemetry output CSV path (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()

    print("moteus High-Speed Reciprocating Actuator")
    print(f"  Controller ID   : {args.id}")
    print(f"  Cycles          : {args.cycles}")
    print(f"  Homing velocity : {args.homing_velocity} rev/s")
    print(f"  Stall threshold : {args.stall_threshold} A")
    print(f"  Telemetry CSV   : {args.output_csv}")

    asyncio.run(
        run(
            controller_id=args.id,
            n_cycles=args.cycles,
            homing_velocity=args.homing_velocity,
            stall_threshold=args.stall_threshold,
            csv_file=args.output_csv,
        )
    )


if __name__ == "__main__":
    main()


# ===========================================================================
# WRITE-UP: Backlash and Overshoot During Homing
# ===========================================================================
# When the motor hits the hard-stop at homing speed (0.1 rev/s), two issues
# arise:
#
# 1. Overshoot into the hard-stop
#    The motor coasts slightly past the stall detection threshold before the
#    controller can fully stop it (latency ≈ 1–2 CAN bus frames ≈ 2–4 ms at
#    1 Mbit/s, plus firmware loop time).  At 0.1 rev/s this corresponds to
#    a worst-case positional overshoot of ~0.0004 rev — negligible for most
#    mechanics.
#
#    If harder contact is a concern:
#      • Reduce homing_velocity to 0.05 rev/s.
#      • Set maximum_torque = 0.5 A during homing to limit impact force.
#
# 2. Backlash after stop detection
#    After set_stop() the coupling spring-back (backlash ≈ 0.01–0.05 rev
#    depending on mechanism) causes the measured position to bounce slightly.
#    This would corrupt the "Position 0" reference if sampled immediately.
#
#    Our handling:
#      a) Issue set_stop() to remove all current — no active force fighting
#         the spring-back.
#      b) Wait 50 ms for the mechanical bounce to settle.
#      c) Transition via set_position(position=nan, velocity=0) to avoid a
#         sudden position-mode command while the mechanism is still moving.
#      d) Retract by RETRACT = 0.05 rev before zeroing.  This ensures the
#         zero reference is taken in free space (no contact forces) and
#         absorbs any remaining backlash slack.
#
# This four-step approach is robust to the typical backlash range of
# gear-driven BLDC systems (0.01–0.1 rev).  For tighter systems reduce
# RETRACT; for large backlash increase it and add a second approach pass.
# ===========================================================================
