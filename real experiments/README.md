# Real Experiments for Section 4.5

This folder collects the code entry points for the four real-robot validation experiments planned for Section 4.5. The files are copied from the verified live experiment scripts and should be migrated to Ubuntu for execution on the physical robot.

Important: these scripts still contain the CoppeliaSim Remote API execution interface. Before running real experiments, replace the simulator communication layer with the real robot driver interface used on Ubuntu. Do not use simulator outputs as real-robot results.

## Experiment Set

### 1. Clean Repetitive Tracking

Folder:

```text
01_clean_repetitive_tracking/
```

Main script:

```text
run_clean_repetitive_tracking.py
```

Purpose:

- verify clean repetitive position tracking on the real robot;
- record desired and measured end-effector trajectories;
- record position tracking error, joint angles, joint velocities, and joint drift relative to the initial joint configuration.

Recommended main trajectory:

```text
circle
```

The file `final_live_best_tau0005.json` stores the tuned parameter sets from the previous live experiments. Use the `circle` entry first unless a different trajectory is intentionally selected.

### 2. With vs Without Drift-Free Constraint

Folder:

```text
02_drift_free_ablation/
```

Scripts:

```text
run_with_drift_free.py
run_without_drift_free.py
```

Purpose:

- compare the proposed drift-free formulation with a version where the drift-free mechanism is removed or disabled;
- keep the same robot, trajectory, sampling period, initial joint configuration, and controller gains as far as possible;
- report final joint drift, accumulated joint drift norm, and position tracking error.

This experiment is the key ablation for showing why the drift-free constraint is necessary in repetitive tracking.

### 3. Real-Time Feasibility

Folder:

```text
03_real_time_feasibility/
```

Script:

```text
benchmark_step_time.py
```

Purpose:

- measure computation time per control cycle on the Ubuntu real-experiment machine;
- report mean, median, 95th percentile, 99th percentile, maximum computation time, and control deadline miss count;
- verify whether the controller update usually finishes within the sampling period.

For a sampling period of

```text
tau = 0.005 s
```

the basic deadline is

```text
5 ms per control cycle
```

The computation time should be measured around the controller update only, not around plotting or file writing.

### 4. Mild Robustness

Folder:

```text
04_mild_robustness/
```

Main script:

```text
run_mild_disturbance.py
```

Purpose:

- test whether the real robot can retain usable tracking and drift-free behavior under a mild disturbance;
- suggested disturbance options include a small desired-velocity disturbance, a small command-level disturbance, or a small measurement-noise injection;
- avoid unsafe physical disturbance unless the supervisor approves the setup.

Recommended first test:

```text
n(t) = 0.5t
```

or a safer software-level perturbation with comparable magnitude.

## Ubuntu Migration Checklist

1. Create a Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
```

2. Replace the CoppeliaSim interface.

Search for these functions and calls in each script:

```text
connect_to_coppeliasim
simxSetJointTargetPosition
read_joint_positions_fast
read_object_position_fast
reset_simulation_with_toolbar_equivalent
```

Replace them with the real robot interface:

```text
connect_to_robot
send_joint_position_command or send_joint_velocity_command
read_joint_positions
read_end_effector_position
reset_or_move_to_initial_configuration
```

3. Keep the measurement definitions fixed.

Use the same definitions across all four experiments:

```text
position error = measured end-effector position - desired end-effector position
joint drift = measured joint position - initial measured joint position
computation time = elapsed wall-clock time of one controller update
```

4. Run low-risk tests first.

Recommended order:

```text
1. one short clean circle trial
2. full clean circle trial
3. with/without drift-free ablation
4. real-time computation timing
5. mild robustness trial
```

5. Save raw data.

For each run, save at least:

```text
time
desired_position
measured_position
position_error
joint_position
joint_velocity or command increment
joint_drift
computation_time_per_cycle
controller_parameters
initial_joint_configuration
```

## Safety Notes

- Start with reduced trajectory amplitude and reduced speed.
- Confirm joint limits and velocity limits before enabling the robot.
- Keep an emergency stop available.
- Do not run the no-drift-free ablation at full scale until the clean drift-free run is stable.
- Do not apply physical disturbance during the first validation pass.

## Section 4.5 Writing Logic

Use the real experiment results in this order:

1. clean repetitive tracking;
2. with vs without drift-free constraint;
3. real-time feasibility;
4. mild robustness.

The section should support one claim: JDF-DLCCZNN can retain position tracking accuracy and joint drift-free behavior in a real control chain under sampled execution.
