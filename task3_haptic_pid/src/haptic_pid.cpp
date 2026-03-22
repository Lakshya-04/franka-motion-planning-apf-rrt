/**
 * haptic_pid.cpp
 * ==============
 * Implementation of the PID class declared in haptic_pid.hpp.
 * Haptic Dial PID — anti-windup (clamping), derivative LPF, detent helper
 *
 * Build (via CMake — preferred):
 *   cmake -S . -B build && cmake --build build
 *
 * Build (manual, two translation units):
 *   g++ -std=c++17 -Wall -Wextra -O2 haptic_pid.cpp main.cpp -o haptic_pid_demo
 */

#include "haptic_pid.hpp"

#include <algorithm>  // std::clamp
#include <cmath>      // M_PI

// ===========================================================================
// PID constructor
// ===========================================================================

/**
 * @brief Initialise all tuning parameters and pre-compute the IIR alpha
 *        coefficient for the derivative low-pass filter.
 *
 * @details The IIR alpha is derived from the bilinear-transform approximation:
 *   @code
 *   tau   = 1 / (2π · f_c)
 *   alpha = dt / (dt + tau)
 *   @endcode
 *   A positive @p lpf_cutoff_hz pre-computes a meaningful alpha; zero (or
 *   negative) bypasses filtering entirely (alpha = 1.0 collapses the IIR
 *   recurrence to d_filtered = d_raw every cycle).
 *
 * @note All integrator and derivative state members are zero-initialised in
 *       the class definition, so no explicit reset is needed at construction.
 */
PID::PID(float kp, float ki, float kd, float dt,
         float out_min, float out_max, float lpf_cutoff_hz)
    : kp_(kp),
      ki_(ki),
      kd_(kd),
      dt_(dt),
      out_min_(out_min),
      out_max_(out_max) {
  if (lpf_cutoff_hz > 0.0f) {
    // τ = RC time constant of an equivalent continuous-time first-order filter.
    // Smaller τ (higher f_c) → alpha closer to 1 → less smoothing.
    const float tau = 1.0f / (2.0f * static_cast<float>(M_PI) * lpf_cutoff_hz);
    // Bilinear-transform alpha: maps the continuous-time RC pole to discrete time.
    lpf_alpha_ = dt / (dt + tau);
  } else {
    lpf_alpha_ = 1.0f;  // pass-through — d_filtered tracks d_raw exactly
  }
}

// ===========================================================================
// PID::update
// ===========================================================================

/**
 * @brief Compute one PID output sample using clamping anti-windup and an
 *        IIR-filtered derivative term.
 *
 * @details Execution order each cycle:
 *   1. Compute position error.
 *   2. Scale by Kp for the proportional term.
 *   3. Finite-difference the *error* (not setpoint) and pass through IIR filter.
 *   4. Sum P+D to determine the integrator's available headroom.
 *   5. Accumulate the integral and clamp it to that headroom (anti-windup).
 *   6. Apply a final hard clamp on total output as a floating-point safety net.
 *   7. Save error for the next cycle's finite-difference.
 *
 * @note Differentiating the error (step 3) rather than the setpoint avoids
 *       a "derivative kick" when the setpoint jumps discretely to a new detent.
 */
float PID::update(float setpoint, float measured) {
  // 1. Error: positive when measured position lags behind the setpoint.
  const float error = setpoint - measured;

  // 2. Proportional term: direct linear restoring force toward the setpoint.
  const float p_term = kp_ * error;

  // 3. IIR-filtered derivative on the error signal.
  //    Differentiating the error (not the setpoint) avoids derivative kick
  //    when the setpoint jumps step-wise to a new detent position.
  const float d_raw = (error - prev_error_) / dt_;
  // Recurrence: blends the new raw sample with the previous filtered value.
  // alpha near 1 → more responsive; alpha near 0 → heavier smoothing.
  d_filtered_ = lpf_alpha_ * d_raw + (1.0f - lpf_alpha_) * d_filtered_;
  const float d_term = kd_ * d_filtered_;

  // 4. P+D combined — determines how much of [out_min, out_max] is already
  //    consumed before adding the integral, i.e. the integrator's headroom.
  const float output_pd = p_term + d_term;

  // 5. Anti-windup: clamp integrator to the headroom left after P+D.
  //    Clamping to [out_min-output_pd, out_max-output_pd] guarantees that
  //    (output_pd + integral_) stays within [out_min, out_max].
  //    At a hard-stop the output is saturated, so the integrator is frozen —
  //    no wind-up occurs and there is no torque lurch on release.
  const float i_candidate = integral_ + ki_ * error * dt_;
  integral_ = std::clamp(i_candidate,
                          out_min_ - output_pd,
                          out_max_ - output_pd);

  // 6. Total output — hard clamp as a safety net against floating-point drift.
  const float output = std::clamp(output_pd + integral_, out_min_, out_max_);

  // 7. Store error for the next cycle's finite-difference derivative.
  prev_error_ = error;

  return output;
}

// ===========================================================================
// PID::reset
// ===========================================================================

/**
 * @brief Clear all integrator and derivative state back to zero.
 *
 * @details Call when re-activating a paused axis or after a large setpoint
 *          relocation to prevent stale values from producing an unexpected
 *          torque transient on the first update() call after resumption.
 */
void PID::reset() {
  integral_   = 0.0f;  // discard accumulated integral wind-up
  prev_error_ = 0.0f;  // prevent a spurious derivative spike on next update
  d_filtered_ = 0.0f;  // flush the IIR filter memory
}
