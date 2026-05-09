# Method3 TVQP ELNCP PCG Solver

This project converts the Method3 position and drift objective into a time varying box QP and replaces the original post solve velocity clipping with an ELNCP styled KKT boundary treatment. The reduced free variable SPD subsystem is solved by warm started PCG at every sampling instant.

## Layout

- `scripts/run_offline.py`: complete offline experiment script.
- `scripts/run_live.py`: complete CoppeliaSim live experiment script.
- `docs/method3_tvqp_elncp_pcg_report.md`: formula derivation, experiment results, and comparison.
- `results/standard_heart20_default/offline`: standard 20 s offline results.
- `results/standard_heart10_default/live`: standard 10 s live CoppeliaSim results.

## Retained Baseline Parameters

$$
k_p=198,\quad w_p=15000,\quad w_d=5\times 10^{-4},\quad \epsilon_Q=10^{-10},\quad \tau=0.005.
$$

Solver specific settings:

$$
k_{\rm pcg}^{\max}=8,\quad k_{\rm act}^{\max}=16,\quad \varepsilon_{\rm ncp}=10^{-10}.
$$

## Main Results

The standard 20 s offline run gives:

$$
\bar e_p=2.0433668773123088\times 10^{-4}\ {\rm m},\quad
e_p(T)=6.079975503728618\times 10^{-7}\ {\rm m},\quad
\|\Delta q(T)\|_2=4.6671803940884594\times 10^{-7}\ {\rm rad}.
$$

The standard 10 s live run gives:

$$
\bar e_p=1.302305246026711\times 10^{-4}\ {\rm m},\quad
e_p(T)=5.8159545931849617\times 10^{-5}\ {\rm m},\quad
\|\Delta q(T)\|_2=4.170262650664861\times 10^{-4}\ {\rm rad}.
$$

## Run Commands

```powershell
cd C:\Users\lyx\Desktop\sci\experiments\spd_box_qp_pcg\project_method3_tvqp_elncp_pcg_solver
python .\scripts\run_offline.py --offline-duration 20 --output-root .\results\standard_heart20_default
python .\scripts\run_live.py --duration 10 --output-root .\results\standard_heart10_default
```
