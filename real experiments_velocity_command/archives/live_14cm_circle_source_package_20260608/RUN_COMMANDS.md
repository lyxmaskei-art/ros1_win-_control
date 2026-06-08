# Run Commands

All commands were run from `C:\Users\lyx\Desktop\sci`.

## 14 cm `circle_periodic`

```powershell
python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_clean_repetitive_tracking_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle_periodic --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --drift-gain 50 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_old_params_20260608\01_clean' --skip-plots

python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_with_drift_free_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle_periodic --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --drift-gain 50 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_old_params_20260608\02_with_drift_free' --skip-plots

python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_without_drift_free_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle_periodic --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_old_params_20260608\02_without_drift_free' --skip-plots

python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_mild_disturbance_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle_periodic --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --drift-gain 50 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --internal-disturbance triangle --disturbance-scale 1.0 --command-delta-limit 0.0 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_old_params_20260608\04_mild_robustness_triangle' --skip-plots
```

## 14 cm ordinary `circle`

```powershell
python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_clean_repetitive_tracking_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --drift-gain 50 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_circle_old_params_20260608\01_clean' --skip-plots

python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_with_drift_free_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --drift-gain 50 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_circle_old_params_20260608\02_with_drift_free' --skip-plots

python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_without_drift_free_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_circle_old_params_20260608\02_without_drift_free' --skip-plots

python 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\servo_like_live\run_mild_disturbance_servo_like.py' --experiment single --method method2_dlccznn --trajectory-name circle --duration 20 --offline-duration 20 --trajectory-period 10 --heart-scale 0.0175 --tau 0.005 --task-gain 340 --drift-gain 50 --solver-gamma 4352 --activation-power 0.8 --activation-exp-clip 4.0 --drift-feedback-mode nonlinear --dlccznn-inner-steps 60 --internal-disturbance triangle --disturbance-scale 1.0 --command-delta-limit 0.0 --output-root 'C:\Users\lyx\Desktop\sci\real experiments_velocity_command\paper_runs\live_diameter14cm_circle_old_params_20260608\04_mild_robustness_triangle' --skip-plots
```

