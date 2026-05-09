# Method 3c Formula and Code Mapping

## Outer objective

Method 3c keeps the Method 3a sampled objective:

$$
\min_{\dot q_k}\;
\alpha \left\| e_p(k) + \tau \bigl(J_k \dot q_k - \dot r_d(k) + k_p e_p(k)\bigr) \right\|_2^2
+ \beta \left\| e_d(k) + \tau \dot q_k \right\|_2^2
+ \varepsilon \left\| \dot q_k \right\|_2^2 .
$$

This gives the quadratic root problem

$$
Q_k \dot q_k + c_k = 0,
$$

with

$$
Q_k = \alpha J_k^\top J_k + \beta I + \varepsilon I,
$$

$$
c_k = \alpha J_k^\top \left(\left(\frac{1}{\tau} + k_p\right)e_p(k) - \dot r_d(k)\right) + \beta \frac{e_d(k)}{\tau}.
$$

## Strict DLCCZNN mapping

Starting from the residual

$$
E(t) = Q(t) z(t) + c(t),
$$

define the energy

$$
v(t) = \frac{1}{2}\|E(t)\|_2^2 .
$$

Following the paper, enforce

$$
\dot v(t) = -\mu \,\phi(v(t)),
\qquad
\phi(v) = v^r e^{|v|},
\qquad 0 < r < 1 .
$$

Because

$$
\dot E(t) = \dot Q(t) z(t) + Q(t)\dot z(t) + \dot c(t),
$$

we have

$$
\dot v(t) = E(t)^\top \dot E(t)
= E(t)^\top \bigl(\dot Q(t) z(t) + Q(t)\dot z(t) + \dot c(t)\bigr).
$$

Hence the strict continuous-time DLCCZNN style update is

$$
\dot z(t)
=
-\frac{Q(t)^\top E(t)}
{\|Q(t)^\top E(t)\|_2^2 + \lambda}
\Bigl(E(t)^\top\bigl(\dot Q(t) z(t) + \dot c(t)\bigr) + \mu \phi(v(t))\Bigr).
$$

Applying forward Euler gives

$$
z_{\ell+1}
=
z_\ell
- h
\frac{Q_\ell^\top E_\ell}
{\|Q_\ell^\top E_\ell\|_2^2 + \lambda}
\Bigl(E_\ell^\top\bigl(\dot Q_\ell z_\ell + \dot c_\ell\bigr) + \mu \phi(v_\ell)\Bigr).
$$

## Executable code path

The integrated executable file currently keeps the same residual definition and DLCCZNN direction, but uses the numerically stable frozen-step implementation

$$
E_\ell = Q_k z_\ell + c_k,
\qquad
v_\ell = \frac{1}{2}\|E_\ell\|_2^2,
$$

$$
z_{\ell+1}
=
\Pi_{[\xi^-,\xi^+]}\left(
z_\ell
- h
\frac{Q_k^\top E_\ell}
{\|Q_k^\top E_\ell\|_2^2 + \lambda}
\mu \phi(v_\ell)
\right),
$$

where $h = \tau / N_s$ and $N_s$ is the number of inner substeps.

This was kept because the fully time-varying finite-difference version of $\dot Q$ and $\dot c$ was numerically unstable in the current UR3e repeated-kinematics setting. The experiment results therefore need to be interpreted honestly:

1. the derivation route is strict and clear;
2. the stable code path is a frozen-step DLCCZNN approximation of that route; and
3. for Method 3c this approximation preserves position accuracy but does not preserve the original drift-free recovery effect.

## Code mapping

- `build_qp_terms(...)`: forms $Q_k$ and $c_k$
- `run_dlccznn_inner_solver(...)`: updates the internal velocity state $z_\ell$
- `Method3cController.step(...)`: combines the outer control logic, dynamic bounds, and the DLCCZNN inner update
