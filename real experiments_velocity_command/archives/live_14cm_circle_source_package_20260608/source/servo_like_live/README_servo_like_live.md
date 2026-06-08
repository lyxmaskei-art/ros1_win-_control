# Servo-like CoppeliaSim Live Diagnostics

This folder keeps the original kinematic live scripts untouched and adds
servo-like CoppeliaSim live variants.

## Finding

The current UR3e CoppeliaSim scene does not move under pure
`setJointTargetVelocity` commands, even when the joints are configured as
dynamic velocity-control joints. A direct ZMQ probe commanded joint 2 with
`0.2 rad/s` for `30` steps at `tau=0.005 s`; expected motion was `0.03 rad`,
but measured motion was `0 rad`.

The same scene does move under dynamic target-position control via
`setJointTargetPosition`. Therefore the fixed servo-like implementation is a
`servoj`-equivalent path:

```text
controller qdot(k) -> q_target(k+1) = q_feedback(k) + qdot(k) * tau
                   -> sim.setJointTargetPosition(q_target)
```

It deliberately does not use direct `setJointPosition(current + qdot * tau)`.

## Current 20 s Results

All runs use `circle_periodic`, `tau=0.005`, `inner_steps=10`,
`task_gain=600`, `solver_gamma=104`, `activation_power=0.85`, and
`theta_dot_limit=0.28`.

| run | mean position error (m) | max position error (m) | final drift norm (rad) |
| --- | ---: | ---: | ---: |
| 01 clean, drift_gain=5 | 4.090330941e-4 | 5.157209246e-4 | 1.598668447e-2 |
| 02 with drift-free, drift_gain=5 | 4.090330941e-4 | 5.157209246e-4 | 1.598668447e-2 |
| 02 without drift-free | 4.326788783e-4 | 5.161103314e-4 | 7.306623942e-1 |
| 04 mild robustness, drift_gain=10 | 3.947391428e-4 | 5.773659525e-4 | 8.089493841e-3 |

These are real CoppeliaSim servo-target executions, not kinematic direct joint
position writes.
