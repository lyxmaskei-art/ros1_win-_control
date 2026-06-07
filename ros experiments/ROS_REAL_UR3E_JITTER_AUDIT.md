# ROS real UR3e jitter audit

Date: 2026-06-07

## Main diagnosis

The current real-robot path is not only a gain-tuning problem. The uploaded
UR3e evidence shows `/joint_states` at about 452-459 Hz and stable 5 ms outer
loop timing, so feedback rate is not the main fault in this run set. The
strongest fault signal is execution-layer lag: in `position_array` and
`joint_trajectory` runs, commanded joint velocity p95 was about 0.89 rad/s but
feedback velocity p95 was only about 0.08-0.12 rad/s. `velocity_array` finally
made commanded and feedback velocities comparable, but still needed stricter
acceleration and jerk shaping.

The README's low-shake command is therefore the right direction:

```bash
python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --real-safe-preset \
  --tau 0.005
```

`--real-safe-preset` fills unset values with `--tcp-offset 0,0,0.145`,
`--task-gain 240`, `--solver-gamma 30`, `--drift-gain 2`,
`--theta-dot-limit 0.3`, `--max-command-accel 8`,
`--max-command-jerk 80`. Explicit CLI values still override the preset.

If this still shakes, do not only keep lowering gains. First verify whether the
published target sequence, ROS loop timing, or robot feedback is the noisy part.

## Code changes in this branch

`ros_real_preflight_check.py` adds a non-moving hardware preflight:

- reads `/joint_states` without commanding motion
- checks `/controller_manager/list_controllers` when available
- records published topic types for the configured command topics
- records ROS master subscriber nodes for the configured command topics
- ranks `joint_trajectory`, `velocity_array`, and `position_array` by subscriber,
  controller-state, and topic compatibility
- samples `/joint_states` arrival rate and ROS stamp age
- writes `preflight_verdict.status` and concrete fail/warn issues

`_ros_real_ur3e.py` now writes cycle-level diagnostics for every ROS real run:

- `ros_real_cycle_diagnostics.csv`
- `ros_real_cycle_diagnostics.npz`
- `ros_real_cycle_diagnostics_summary.json`
- the same summary is embedded under `ros_real_cycle_diagnostics` in the main
  experiment summary JSON

The diagnostics include:

- actual loop period
- whole-cycle work time
- joint-state read time and latest state age
- controller compute time
- publish time
- sleep time
- command step norm
- command velocity norm
- command acceleration norm
- command jerk norm
- raw command velocity norm before command shaping
- feedback velocity norm
- optional measured TCP pose age
- optional measured TCP versus local-FK error norm
- whole-loop deadline misses
- loop-period overruns

Run the analyzer after each real trial:

```bash
python3 analyze_ros_real_diagnostics.py <run-directory>
```

It writes `ros_real_diagnostic_analysis.json` and
`ros_real_diagnostic_analysis.md` next to the cycle diagnostic CSV.

## How to interpret the next run

0. Before commanding motion, run:

   ```bash
   python3 ros_real_preflight_check.py \
     --real-safe-preset \
     --command-mode joint_trajectory \
     --trajectory-command-topic /scaled_pos_joint_traj_controller/command \
     --require-pass
   ```

   If `preflight_verdict.status` is `fail`, do not command motion; follow the
   issue action. If `joint_state_rate_probe.median_rate_hz` is far below 200 Hz
   while `--tau 0.005`, test `--tau 0.01` or fix the driver feedback rate before
   changing controller gains. If `command_mode_recommendations` marks
   `joint_trajectory` unavailable, use the listed subscribed controller instead.

1. If `controller_step_s` is small but `cycle_period_s` or `state_age_s` has high
   p95/p99 values, the problem is ROS/driver timing rather than the DLCCZNN math.

2. If `command_accel_norm_rad_s2` has spikes when the robot shakes, enable or
   lower `--max-command-accel` and reduce `--theta-dot-limit`.

3. If `Command velocity greatly exceeds feedback velocity` appears, the outer
   controller is producing velocity-like increments much faster than the real
   controller follows. Prefer `velocity_array` or a UR servo/RTDE velocity path with accel/jerk limits. If a
   trajectory controller must be used for a real-time test, keep single-point
   commands and treat persistent lag as an interface limitation.

4. If local-FK tracking error looks good while the tool visibly shakes, the
   current metric is insufficient. Pass a measured TCP pose source when
   available:

   ```bash
   --tcp-pose-topic /tool_pose
   ```

   The topic must be `geometry_msgs/PoseStamped` in the same frame as the local
   DH model. With this option, `mean_position_error_m` uses measured TCP pose,
   while `mean_fk_position_error_m` stays as the local FK reference. Large
   `tcp_fk_error_norm_m` means TCP offset/frame calibration must be fixed before
   retuning gains.

5. If timing and command smoothness are acceptable but shake remains, run an
   interface A/B test:

   ```bash
   python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
     --experiment single \
     --method method2_dlccznn \
     --trajectory-name circle \
     --real-safe-preset \
     --command-mode joint_trajectory \
     --trajectory-command-topic /scaled_pos_joint_traj_controller/command \
     --trajectory-command-duration 0.02 \
     --tau 0.005
   ```

    This uses single-point `trajectory_msgs/JointTrajectory` instead of the default
   `std_msgs/Float64MultiArray` position-array stream. It keeps the real-time
   single-command semantics; use it only when that trajectory controller is loaded and subscribed.

   If the robot also has a joint-group velocity controller, run a second
   interface test that matches the algorithm output more directly:

   ```bash
   python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
     --experiment single \
     --method method2_dlccznn \
     --trajectory-name circle \
     --duration 5 \
     --real-safe-preset \
     --command-mode velocity_array \
     --velocity-command-topic /joint_group_vel_controller/command \
     --tau 0.005
   ```

   In this mode the live loop publishes the limited `theta_dot_next` directly.
   The interface attempts to publish zero velocity on normal completion, stack
   unwinding after exceptions, ROS shutdown, and process exit. Keep the first
   test short and keep the emergency stop available.

   After the interface tests, compare the run directories:

   ```bash
   python3 compare_ros_real_runs.py <position_array_run> <joint_trajectory_run> <velocity_array_run>
   ```

   The comparison ranks modes by high-severity diagnostic findings, timing
   stability, feedback age, command acceleration, tracking error, and drift.

6. Package the evidence for review:

   ```bash
   python3 collect_ros_real_artifacts.py \
     <position_array_run> <joint_trajectory_run> <velocity_array_run> \
     --output-zip ros_real_triage.zip
   ```

   The zip contains preflight reports, summary JSON, cycle diagnostics, analyzer
   outputs, a manifest, and an A/B comparison. Use `--include-heavy` only when
   history arrays or figures are needed.

## Real-time-control constraint

Multi-point trajectory windows are retained only as an explicit A/B diagnostic, not as the recommended real-time hardware validation path. The default remains one command per control cycle.

## Safer trial sequence

Start with a short low-risk run:

```bash
python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --duration 5 \
  --real-safe-preset \
  --command-mode velocity_array \
  --velocity-command-topic /joint_group_vel_controller/command \
  --tcp-offset 0,0,0.145 \
  --task-gain 160 \
  --solver-gamma 20 \
  --drift-gain 1 \
  --theta-dot-limit 0.25 \
  --max-command-accel 6 \
  --max-command-jerk 60 \
  --tau 0.005
```

Then move toward the README values only after the diagnostic summary shows stable
loop timing and smooth command acceleration.
