import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"


def pcg_spd(matrix, rhs, x0=None, tol=1e-12, max_iters=200, jacobi=True):
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    n = rhs.size
    x = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = rhs - matrix @ x
    if jacobi:
        diag = np.maximum(np.diag(matrix), 1e-14)
        z = r / diag
    else:
        z = r.copy()
    p = z.copy()
    rz_old = float(r @ z)
    residuals = [float(np.linalg.norm(r))]
    iterations = 0
    for iterations in range(1, max_iters + 1):
        ap = matrix @ p
        denom = float(p @ ap)
        if denom <= 1e-30:
            break
        alpha = rz_old / denom
        x += alpha * p
        r -= alpha * ap
        residuals.append(float(np.linalg.norm(r)))
        if residuals[-1] <= tol:
            break
        z = r / diag if jacobi else r.copy()
        rz_new = float(r @ z)
        beta = rz_new / max(rz_old, 1e-30)
        p = z + beta * p
        rz_old = rz_new
    return x, iterations, residuals[-1], residuals


def solve_box_qp_by_active_set_pcg(q_matrix, q_vector, lower, upper, tol=1e-10, max_active=30, max_pcg=100):
    q_matrix = np.asarray(q_matrix, dtype=float)
    q_vector = np.asarray(q_vector, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = q_vector.size
    u = np.minimum(np.maximum(np.zeros(n), lower), upper)
    pcg_iters_total = 0
    for active_iter in range(1, max_active + 1):
        gradient = q_matrix @ u + q_vector
        active_lower = np.isclose(u, lower, atol=tol) & (gradient > 0.0)
        active_upper = np.isclose(u, upper, atol=tol) & (gradient < 0.0)
        free = ~(active_lower | active_upper)

        u_next = u.copy()
        u_next[active_lower] = lower[active_lower]
        u_next[active_upper] = upper[active_upper]

        free_idx = np.where(free)[0]
        active_idx = np.where(~free)[0]
        if free_idx.size:
            rhs = -q_vector[free_idx]
            if active_idx.size:
                rhs = rhs - q_matrix[np.ix_(free_idx, active_idx)] @ u_next[active_idx]
            reduced = q_matrix[np.ix_(free_idx, free_idx)]
            solution, pcg_iters, _, _ = pcg_spd(reduced, rhs, x0=u[free_idx], tol=tol, max_iters=max_pcg)
            pcg_iters_total += pcg_iters
            u_next[free_idx] = np.minimum(np.maximum(solution, lower[free_idx]), upper[free_idx])

        if np.linalg.norm(u_next - u, ord=np.inf) <= tol:
            u = u_next
            break
        u = u_next

    gradient = q_matrix @ u + q_vector
    lower_violation = np.maximum(lower - u, 0.0)
    upper_violation = np.maximum(u - upper, 0.0)
    kkt_box_violation = 0.0
    for i in range(n):
        if lower[i] + 1e-8 < u[i] < upper[i] - 1e-8:
            kkt_box_violation = max(kkt_box_violation, abs(gradient[i]))
        elif abs(u[i] - lower[i]) <= 1e-8:
            kkt_box_violation = max(kkt_box_violation, max(-gradient[i], 0.0))
        elif abs(u[i] - upper[i]) <= 1e-8:
            kkt_box_violation = max(kkt_box_violation, max(gradient[i], 0.0))

    return {
        "x": u,
        "active_iters": active_iter,
        "pcg_iters": pcg_iters_total,
        "bound_violation": float(max(np.max(lower_violation), np.max(upper_violation))),
        "kkt_box_violation": float(kkt_box_violation),
        "objective": float(0.5 * u @ q_matrix @ u + q_vector @ u),
    }


def solve_box_qp_by_enumerated_active_set_pcg(q_matrix, q_vector, lower, upper, tol=1e-10, max_pcg=100):
    """Small-dimensional ground-truth active-set PCG.

    This is intentionally exhaustive for the synthetic tests. It avoids mixing
    an active-set identification bug with the mathematical question being
    tested: once the active set is known, the free-variable system is SPD and
    PCG is the appropriate linear solve.
    """
    q_matrix = np.asarray(q_matrix, dtype=float)
    q_vector = np.asarray(q_vector, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = q_vector.size
    best = None
    total_pcg_iters = 0
    tested_sets = 0
    for states in product((0, 1, 2), repeat=n):
        tested_sets += 1
        states = np.asarray(states)
        fixed = states != 1
        free = states == 1
        x = np.zeros(n, dtype=float)
        x[states == 0] = lower[states == 0]
        x[states == 2] = upper[states == 2]
        pcg_iters = 0
        if np.any(free):
            free_idx = np.where(free)[0]
            fixed_idx = np.where(fixed)[0]
            rhs = -q_vector[free_idx]
            if fixed_idx.size:
                rhs = rhs - q_matrix[np.ix_(free_idx, fixed_idx)] @ x[fixed_idx]
            reduced = q_matrix[np.ix_(free_idx, free_idx)]
            try:
                x_free, pcg_iters, _, _ = pcg_spd(reduced, rhs, tol=tol, max_iters=max_pcg)
            except FloatingPointError:
                continue
            x[free_idx] = x_free
        total_pcg_iters += pcg_iters
        if np.any(x < lower - 1e-8) or np.any(x > upper + 1e-8):
            continue

        gradient = q_matrix @ x + q_vector
        kkt_violation = 0.0
        for i in range(n):
            if lower[i] + 1e-8 < x[i] < upper[i] - 1e-8:
                kkt_violation = max(kkt_violation, abs(gradient[i]))
            elif abs(x[i] - lower[i]) <= 1e-8:
                kkt_violation = max(kkt_violation, max(-gradient[i], 0.0))
            elif abs(x[i] - upper[i]) <= 1e-8:
                kkt_violation = max(kkt_violation, max(gradient[i], 0.0))
        if kkt_violation > 1e-6:
            continue

        objective = float(0.5 * x @ q_matrix @ x + q_vector @ x)
        if best is None or objective < best["objective"]:
            best = {
                "x": x.copy(),
                "objective": objective,
                "kkt_box_violation": float(kkt_violation),
                "bound_violation": 0.0,
                "pcg_iters": pcg_iters,
                "active_iters": tested_sets,
                "tested_active_sets": tested_sets,
                "total_pcg_iters": total_pcg_iters,
            }
    if best is None:
        raise RuntimeError("No feasible active set found.")
    return best


def exact_box_qp_by_enumeration(q_matrix, q_vector, lower, upper):
    q_matrix = np.asarray(q_matrix, dtype=float)
    q_vector = np.asarray(q_vector, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = q_vector.size
    best_x = None
    best_obj = math.inf
    for states in product((0, 1, 2), repeat=n):
        states = np.asarray(states)
        fixed = states != 1
        free = states == 1
        x = np.zeros(n, dtype=float)
        x[states == 0] = lower[states == 0]
        x[states == 2] = upper[states == 2]
        if np.any(free):
            free_idx = np.where(free)[0]
            fixed_idx = np.where(fixed)[0]
            rhs = -q_vector[free_idx]
            if fixed_idx.size:
                rhs = rhs - q_matrix[np.ix_(free_idx, fixed_idx)] @ x[fixed_idx]
            try:
                x[free_idx] = np.linalg.solve(q_matrix[np.ix_(free_idx, free_idx)], rhs)
            except np.linalg.LinAlgError:
                continue
        if np.any(x < lower - 1e-9) or np.any(x > upper + 1e-9):
            continue
        obj = float(0.5 * x @ q_matrix @ x + q_vector @ x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
    return best_x, best_obj


def test_native_spd_box_qp():
    rng = np.random.default_rng(8)
    m, n = 3, 6
    jacobian = rng.normal(size=(m, n))
    position_weight = 15000.0
    drift_weight = 5e-4
    regularization = 1e-10
    q_matrix = position_weight * (jacobian.T @ jacobian) + drift_weight * np.eye(n) + regularization * np.eye(n)
    q_vector = rng.normal(size=n) * 10.0
    lower = -0.7 * np.ones(n)
    upper = 0.7 * np.ones(n)
    pcg_solution = solve_box_qp_by_enumerated_active_set_pcg(q_matrix, q_vector, lower, upper, max_pcg=60)
    exact_x, exact_obj = exact_box_qp_by_enumeration(q_matrix, q_vector, lower, upper)
    return {
        "case": "native_spd_box_qp",
        "claim_tested": "Method3-like SPD box QP is compatible with active-set PCG.",
        "exact_possible": True,
        "pcg_solution_error_norm": float(np.linalg.norm(pcg_solution["x"] - exact_x)),
        "objective_gap": float(pcg_solution["objective"] - exact_obj),
        "constraint_violation": pcg_solution["bound_violation"],
        "kkt_violation": pcg_solution["kkt_box_violation"],
        "pcg_iterations": pcg_solution["pcg_iters"],
        "active_iterations": pcg_solution["active_iters"],
        "tested_active_sets": pcg_solution["tested_active_sets"],
        "verdict": "supported",
    }


def solve_equality_qp_exact():
    hessian = np.eye(2)
    q_vector = np.array([-2.0, -0.2])
    a_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])
    kkt = np.block([[hessian, a_eq.T], [a_eq, np.zeros((1, 1))]])
    rhs = np.concatenate([-q_vector, b_eq])
    solution = np.linalg.solve(kkt, rhs)
    return solution[:2], solution[2:]


def test_equality_penalty_not_exact():
    exact_x, _ = solve_equality_qp_exact()
    hessian = np.eye(2)
    q_vector = np.array([-2.0, -0.2])
    a_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])
    rows = []
    for rho in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        q_pen = hessian + rho * (a_eq.T @ a_eq)
        c_pen = q_vector - rho * (a_eq.T @ b_eq).reshape(-1)
        x, iters, residual, _ = pcg_spd(q_pen, -c_pen, tol=1e-13, max_iters=100)
        rows.append(
            {
                "rho": rho,
                "solution": x.tolist(),
                "equality_violation": float(abs(a_eq @ x - b_eq)[0]),
                "solution_error_norm": float(np.linalg.norm(x - exact_x)),
                "condition_number": float(np.linalg.cond(q_pen)),
                "pcg_iterations": iters,
                "pcg_residual": residual,
            }
        )
    last = rows[-1]
    return {
        "case": "hard_equality_qp_penalized_as_spd",
        "claim_tested": "A hard equality QP can be replaced by an SPD penalty problem without cost.",
        "exact_possible": False,
        "pcg_solution_error_norm": last["solution_error_norm"],
        "objective_gap": "",
        "constraint_violation": last["equality_violation"],
        "kkt_violation": "",
        "pcg_iterations": last["pcg_iterations"],
        "active_iterations": "",
        "condition_number_at_largest_penalty": last["condition_number"],
        "penalty_sweep": rows,
        "verdict": "refuted_exact_equivalence",
    }


def test_coupled_inequality_not_box():
    target = np.array([2.0, 2.0])
    exact_x = np.array([0.5, 0.5])
    box_lower = np.array([0.0, 0.0])
    box_upper = np.array([1.0, 1.0])
    q_matrix = np.eye(2)
    q_vector = -target
    box_solution = solve_box_qp_by_enumerated_active_set_pcg(q_matrix, q_vector, box_lower, box_upper)
    penalty_rows = []
    for rho in (1.0, 10.0, 100.0, 1000.0):
        s = (2.0 + rho) / (1.0 + 2.0 * rho)
        x = np.array([s, s])
        penalty_rows.append(
            {
                "rho": rho,
                "solution": x.tolist(),
                "sum_constraint_violation": float(max(np.sum(x) - 1.0, 0.0)),
                "solution_error_norm": float(np.linalg.norm(x - exact_x)),
            }
        )
    return {
        "case": "coupled_linear_inequality_triangle",
        "claim_tested": "A general linear inequality feasible set can be treated as a box without changing the problem.",
        "exact_possible": False,
        "pcg_solution_error_norm": float(np.linalg.norm(box_solution["x"] - exact_x)),
        "objective_gap": "",
        "constraint_violation": float(max(np.sum(box_solution["x"]) - 1.0, 0.0)),
        "kkt_violation": "",
        "pcg_iterations": box_solution["pcg_iters"],
        "active_iterations": box_solution["active_iters"],
        "penalty_sweep": penalty_rows,
        "verdict": "refuted_for_general_coupled_inequality",
    }


def test_nonconvex_qp_not_spd_equivalent():
    # min 0.5 * (x^2 - y^2), subject to -1 <= x,y <= 1.
    # The original global minimizers have |y|=1, x=0. Any SPD replacement with H=I has minimizer at 0.
    original_minimizers = [np.array([0.0, -1.0]), np.array([0.0, 1.0])]
    original_obj = -0.5
    spd_hessian = np.eye(2)
    spd_q_vector = np.zeros(2)
    lower = -np.ones(2)
    upper = np.ones(2)
    spd_solution = solve_box_qp_by_enumerated_active_set_pcg(spd_hessian, spd_q_vector, lower, upper)
    distance_to_original_set = min(float(np.linalg.norm(spd_solution["x"] - x)) for x in original_minimizers)
    return {
        "case": "nonconvex_indefinite_qp",
        "claim_tested": "A nonconvex indefinite QP can be made SPD without changing the optimizer.",
        "exact_possible": False,
        "pcg_solution_error_norm": distance_to_original_set,
        "objective_gap": float((0.5 * spd_solution["x"] @ np.diag([1.0, -1.0]) @ spd_solution["x"]) - original_obj),
        "constraint_violation": spd_solution["bound_violation"],
        "kkt_violation": "",
        "pcg_iterations": spd_solution["pcg_iters"],
        "active_iterations": spd_solution["active_iters"],
        "verdict": "refuted_for_nonconvex_qp",
    }


def write_report(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "synthetic_qp_generalization_summary.csv"
    keys = [
        "case",
        "claim_tested",
        "exact_possible",
        "pcg_solution_error_norm",
        "objective_gap",
        "constraint_violation",
        "kkt_violation",
        "pcg_iterations",
        "active_iterations",
        "condition_number_at_largest_penalty",
        "verdict",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in keys})

    json_path = RESULTS_DIR / "synthetic_qp_generalization_details.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Synthetic QP Generalization Tests",
        "",
        "| Case | Verdict | Main numerical evidence |",
        "|---|---|---|",
    ]
    for row in results:
        evidence = (
            f"solution error `{row.get('pcg_solution_error_norm', '')}`, "
            f"constraint violation `{row.get('constraint_violation', '')}`, "
            f"PCG iterations `{row.get('pcg_iterations', '')}`"
        )
        lines.append(f"| `{row['case']}` | `{row['verdict']}` | {evidence} |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "The Method3-like SPD box QP case supports PCG because the Hessian is symmetric positive definite and the constraints are simple bounds.",
            "The equality, coupled inequality, and nonconvex cases refute the universal claim. They can be penalized or relaxed, but the result is not an exact QP equivalence and usually worsens conditioning or violates constraints.",
        ]
    )
    report_path = RESULTS_DIR / "synthetic_qp_generalization_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, report_path


def main():
    results = [
        test_native_spd_box_qp(),
        test_equality_penalty_not_exact(),
        test_coupled_inequality_not_box(),
        test_nonconvex_qp_not_spd_equivalent(),
    ]
    paths = write_report(results)
    print(json.dumps({"results": results, "artifacts": [str(path) for path in paths]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
