# Synthetic QP Generalization Tests

| Case | Verdict | Main numerical evidence |
|---|---|---|
| `native_spd_box_qp` | `supported` | solution error `3.608224830031759e-16`, constraint violation `0.0`, PCG iterations `3` |
| `hard_equality_qp_penalized_as_spd` | `refuted_exact_equivalence` | solution error `4.242428565685446e-05`, constraint violation `5.9997000149802915e-05`, PCG iterations `3` |
| `coupled_linear_inequality_triangle` | `refuted_for_general_coupled_inequality` | solution error `0.7071067811865476`, constraint violation `1.0`, PCG iterations `0` |
| `nonconvex_indefinite_qp` | `refuted_for_nonconvex_qp` | solution error `1.0`, constraint violation `0.0`, PCG iterations `1` |

Interpretation:

The Method3-like SPD box QP case supports PCG because the Hessian is symmetric positive definite and the constraints are simple bounds.
The equality, coupled inequality, and nonconvex cases refute the universal claim. They can be penalized or relaxed, but the result is not an exact QP equivalence and usually worsens conditioning or violates constraints.
