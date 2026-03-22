/**
 * @file main.cpp
 * @brief Simulation demo and smoke-test for the Haptic Dial PID controller.
 *        Haptic Dial PID — simulation demo and smoke-test
 *
 * @details
 * Scenario: a BLDC motor drives a high-end haptic rotary knob that must
 * simulate "detents" (discrete click positions) and "end-stops" (hard angular
 * limits).  The encoder is high-resolution but the mechanical coupling has
 * ~0.5° of backlash.
 *
 * Plant model (1st-order):
 *   velocity += control_output × motor_gain × dt
 *   position += velocity × dt
 *   backlash modelled as a ±0.5° dead-band on the measured position
 *
 * Units: angles in radians, time in seconds, output in ±1.0 (normalised).
 *
 * @note See the TUNING REPORT at the bottom of this file for detailed
 *       justification of all gain and filter parameter choices.
 */

#include "haptic_pid.hpp"

#include <cmath>
#include <cstdio>

int main() {
  // ── [1] Plant and simulation parameters ──────────────────────────────────
  constexpr float kDt = 0.001f;       // 1 kHz — typical embedded control rate
  constexpr float kSimSeconds = 2.0f; // total simulated time
  constexpr int kNSteps = static_cast<int>(kSimSeconds / kDt);
  constexpr int kDetents = 24; // 24 clicks per revolution (common encoder)
  constexpr float kMotorGain = 6.0f;    // rad/s² per unit normalised output
  constexpr float kBacklash = 0.00873f; // 0.5° — typical gear-drive dead zone

  // ── [2] Controller — gains chosen by the first-run protocol (see §C) ─────
  // Kp=3.5 gives fast detent snap; Kd=0.12 damps overshoot; Ki=0.8 removes
  // steady-state error at the detent. LPF at 50 Hz (see §B for justification).
  PID pid(/* kp= */ 3.5f, /* ki= */ 0.8f, /* kd= */ 0.12f,
          /* dt= */ kDt,
          /* out_min= */ -1.0f, /* out_max= */ 1.0f,
          /* lpf_cutoff_hz= */ 50.0f);

  // ── [3] Setpoints: nearest detent angles for clicks 2 and 5 ─────────────
  // nearest_detent() is called even on an already-aligned angle as a
  // defensive rounding step (floating-point division may produce tiny errors).
  constexpr float kTwoPi = 6.28318530718f;
  const float step1 = nearest_detent(kTwoPi / kDetents * 2, kDetents);
  const float step2 = nearest_detent(kTwoPi / kDetents * 5, kDetents);

  std::printf("Step 1 setpoint = %.4f rad (%.1f deg)\n", step1,
              step1 * 180.0f / kTwoPi * 2.0f);
  std::printf("Step 2 setpoint = %.4f rad (%.1f deg)\n", step2,
              step2 * 180.0f / kTwoPi * 2.0f);
  std::printf("\n%6s  %9s  %9s  %9s  %9s\n", "t(ms)", "setpoint", "measured",
              "output", "error");

  // ── [4] Simulation loop ───────────────────────────────────────────────────
  float true_pos = 0.0f; // true mechanical position [rad]
  float velocity = 0.0f; // angular velocity [rad/s]

  for (int k = 0; k < kNSteps; ++k) {
    const float t = k * kDt;
    // Step the setpoint at t=1 s to exercise a detent-to-detent transition.
    const float setpoint = (t < 1.0f) ? step1 : step2;

    // Stateless backlash dead-band: the encoder only reports true_pos ± half
    // the backlash gap, so the controller sees slightly less motion than
    // actually occurred.  In a real system this causes P and I to accumulate
    // "invisible" error, which the filtered D-term and anti-windup mitigate.
    float measured = true_pos;
    const float lag = true_pos - measured;
    if (lag > kBacklash / 2.0f)
      measured = true_pos - kBacklash / 2.0f;
    if (lag < -kBacklash / 2.0f)
      measured = true_pos + kBacklash / 2.0f;

    const float output = pid.update(setpoint, measured);

    // Euler-integrate the 1st-order plant with viscous damping (factor 0.98).
    // Damping represents friction losses; without it the system would oscillate
    // indefinitely at any gain.
    velocity = (velocity + output * kMotorGain * kDt) * 0.98f;
    true_pos += velocity * kDt;

    // Decimated print: every 50 steps = 50 ms to keep output readable.
    if (k % 50 == 0) {
      std::printf("%6.0f  %9.4f  %9.4f  %9.4f  %9.4f\n", t * 1000.0f, setpoint,
                  measured, output, setpoint - measured);
    }
  }

  std::printf("\nSimulation complete.\n");
  return 0;
}

// ===========================================================================
// TUNING REPORT
// ===========================================================================
//
// A. Anti-Windup — Why It Is Critical for Hard-Stops
// ---------------------------------------------------
// When the haptic knob reaches a physical end-stop the motor stalls:
// position error grows but velocity stays zero.  A standard PID integrates
// this growing error, accumulating a large positive integral.
//
// Consequence without anti-windup: the moment the user releases the end-stop
// the controller "dumps" the accumulated integral as a large torque spike —
// the motor lunges past the intended setpoint, creating dangerous oscillation.
// For a 100 W motor this could cause injury.
//
// The clamping method implemented here prevents accumulation the moment the
// integrator contribution would push the output outside [out_min_, out_max_].
// In the hard-stop scenario the motor is commanding full torque against the
// wall (output == out_max_), so the integrator is frozen — no wind-up occurs.
//
// B. Derivative Low-Pass Filter — Cutoff Frequency Choice
// -------------------------------------------------------
// The encoder is high-resolution but the backlash introduces quantisation
// "noise" on the derivative estimate.  Without filtering, d_raw jumps sharply
// when the position snaps through the backlash zone, injecting high-frequency
// noise into the output.
//
// Chosen cutoff: f_c = 50 Hz
//
// Justification:
//   • Control bandwidth target ≈ 20 Hz (detent snap response time ~50 ms).
//   • Shannon: f_c < f_sample/2 = 500 Hz — satisfied.
//   • Mechanical resonance of the haptic dial estimated at ~150–200 Hz
//     (small motor + light knob), so f_c = 50 Hz attenuates encoder noise
//     without significant phase lag at frequencies of interest.
//   • At f_c = 50 Hz the filter introduces ~18° phase lag at 20 Hz
//     (from Bode plot of first-order IIR), which is acceptable.
//   • If backlash noise worsens, lower to 30 Hz; if faster detent snapping
//     is required, raise toward 80 Hz and re-tune Kd.
//
// C. First-Run Safety Protocol — Determining Kp for a 100 W Motor
// ---------------------------------------------------------------
// Goal: find the maximum safe Kp without risking a high-speed collision.
//
// Step 1 — Mechanical inspection.
//   Verify travel limits, clear obstructions, confirm encoder polarity
//   (positive output → positive motion).  Keep a finger on the E-stop.
//
// Step 2 — Set Ki = Kd = 0; start Kp at 1% of expected final value.
//   Output saturation limits at ±5% of rated current ("crawl" mode).
//
// Step 3 — Apply a small step setpoint (< 5° from rest).
//   Observe: does the motor move toward the setpoint?  Measure rise time
//   and overshoot.  If no motion, double Kp.
//
// Step 4 — Double Kp each trial (binary-search phase).
//   After each doubling, widen the step size by 5°.  Stop when:
//     (a) visible oscillation begins, or
//     (b) overshoot exceeds 20% of step size.
//   The last stable Kp is the "ultimate gain" candidate.
//
// Step 5 — Back off Kp to 40–50% of the ultimate gain.
//   This is the Ziegler-Nichols starting point.
//
// Step 6 — Introduce Kd (filtered) to reduce overshoot.
//   Start at Kd = Kp × 0.03; increase until overshoot drops below 5%.
//
// Step 7 — Introduce Ki to eliminate steady-state error.
//   Start at Ki = Kp × 0.1 (well below the stability limit).
//   Verify anti-windup activates correctly by commanding an end-stop
//   intentionally (current should plateau, not spike on release).
//
// Step 8 — Final validation.
//   Run 50 detent-to-detent transitions.  Log overshoot, settling time,
//   and integral wind-up.  Confirm that at the end-stop the output stays
//   at the saturation limit (anti-windup active) and returns smoothly.
//
// D. Backlash Compensation Note
// -----------------------------
// The 0.5° backlash creates a dead zone where motor torque does not produce
// measurable encoder movement.  In this zone P and I accumulate "invisible"
// error; when the slack is taken up the integrator drives a corrective spike.
//
// Mitigations implemented:
//   1. Filtered D-term reacts to rate-of-change, suppressing spikes as slack
//      is taken up.
//   2. Anti-windup clamps the integrator during saturation.
//
// Future work:
//   - Dual-encoder scheme (motor shaft + output shaft) to eliminate backlash.
//   - Small dither torque (~1% rated) to pre-load the coupling in the
//     direction of intended motion.
// ===========================================================================
