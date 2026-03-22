/**
 * haptic_pid.hpp
 * ==============
 * Public interface for the Haptic Dial PID controller.
 * Task 3 — SocialHW Robotics Assessment
 *
 * Design
 * ------
 * PID encapsulates all controller state and tuning parameters.
 * The class is intentionally self-contained (no global state, no singletons)
 * so that multiple independent axes can coexist safely.
 *
 * Anti-windup strategy: CLAMPING
 * --------------------------------
 * The integrator contribution is capped to the headroom left after P+D, so
 * the total output never leaves [out_min, out_max].  At a hard-stop the motor
 * stalls (velocity = 0) while position error grows; the clamping freezes the
 * integrator — no wind-up, no torque lurch on release.
 *
 * Derivative low-pass filter (first-order IIR):
 *   d_filtered[k] = α · d_raw[k]  +  (1 − α) · d_filtered[k-1]
 *   α = dt / (dt + 1 / (2π f_c))
 *
 * Chosen cutoff f_c = 50 Hz (see tuning report in main.cpp for justification).
 */

#pragma once

/**
 * PID — single-axis position controller with clamping anti-windup and
 *       first-order IIR low-pass filter on the derivative term.
 */
class PID {
 public:
  /**
   * Construct and fully initialise a PID controller.
   *
   * @param kp             Proportional gain.
   * @param ki             Integral gain.
   * @param kd             Derivative gain.
   * @param dt             Sample period [s].
   * @param out_min        Lower output saturation limit  (e.g. -1.0).
   * @param out_max        Upper output saturation limit  (e.g. +1.0).
   * @param lpf_cutoff_hz  Derivative low-pass cutoff [Hz]; 0 = bypass.
   */
  PID(float kp, float ki, float kd, float dt,
      float out_min, float out_max,
      float lpf_cutoff_hz = 0.0f);

  /**
   * Compute one PID output sample.  Call once per control cycle.
   *
   * @param setpoint  Desired angular position [rad].
   * @param measured  Measured angular position [rad] (from encoder).
   * @return Normalised control output clamped to [out_min, out_max].
   */
  float update(float setpoint, float measured);

  /** Reset integrator and derivative state (call when re-activating a stopped axis). */
  void reset();

 private:
  // ── Tuning parameters ─────────────────────────────────────────────────────
  float kp_, ki_, kd_, dt_;
  float out_min_, out_max_;
  float lpf_alpha_;

  // ── Internal state (zero-initialised) ─────────────────────────────────────
  float integral_   = 0.0f;
  float prev_error_ = 0.0f;
  float d_filtered_ = 0.0f;
};

/**
 * Returns the nearest detent angle for a given number of detents per revolution.
 *
 * @param angle_rad       Arbitrary angle [rad].
 * @param detents_per_rev Number of discrete click positions per revolution.
 */
inline float nearest_detent(float angle_rad, int detents_per_rev) {
  constexpr float kTwoPi = 6.28318530718f;
  const float step = kTwoPi / static_cast<float>(detents_per_rev);
  // std::round gives the nearest integer → nearest discrete click
  return __builtin_roundf(angle_rad / step) * step;
}
