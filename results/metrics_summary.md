# PID vs MPC — course metrics

| Course | Controller | Success | Collided | Min clear. [m] | Time to goal [s] | Path eff. | Mean track err [m] | Solve mean [ms] | Solve p95 [ms] |
|---|---|---|---|---|---|---|---|---|---|
| slalom | PID | no | YES | 0.14 | — | 0.74 | 0.86 | — | — |
| slalom | MPC | YES | no | 0.40 | 5.60 | 0.93 | 0.16 | 5.1 | 10.6 |
| crossing | PID | YES | no | 0.86 | 5.42 | 0.91 | 0.42 | — | — |
| crossing | MPC | YES | no | 0.39 | 5.54 | 0.98 | 0.15 | 5.4 | 12.1 |
| gauntlet | PID | YES | no | 0.56 | 6.28 | 0.89 | 0.34 | — | — |
| gauntlet | MPC | YES | no | 0.38 | 6.32 | 0.98 | 0.11 | 4.1 | 9.8 |
