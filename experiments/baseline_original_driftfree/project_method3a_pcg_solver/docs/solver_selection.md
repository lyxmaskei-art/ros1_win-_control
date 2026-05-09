# Solver Selection

## Why Method 3a is different

`Method 3a` does not need to solve a KKT saddle-point system like `Method 1/2`.

Its inner problem is

$$
Q_k \dot q_k + c_k = 0,
$$

where `Q_k` is symmetric positive definite after symmetrization and tiny regularization.

This means the problem is not only a generic neural residual-tracking task. It is specifically a small SPD linear system.

## Candidates considered

1. `DLCCZNN / LCCZNN style residual trackers`
   These are structurally inversion-free and attractive from the paper story, but the previous `Method 3c` project already showed that the drift-free effect is sensitive to inner solve accuracy.
2. `PDNN style iterative residual dynamics`
   This is natural for KKT or LVI formulations, but it is not especially well matched to a pure SPD linear solve.
3. `Warm-started PCG`
   This directly matches the SPD structure and uses only matrix-vector products plus vector inner products.

## Why warm-started PCG was chosen

1. `Method 3a` has only `n = 6` joints, so CG reaches the exact solution in at most `n` iterations in exact arithmetic.
2. Consecutive samples are close to each other, so the previous-step solution is a useful warm start.
3. The replacement stays matrix-inversion-free in the online loop while keeping the same outer objective.

## Practical takeaway

For `Method 3a`, the best replacement route found so far is not another neural solver. It is a structure-matched iterative linear solver.
