# MultiJunction: MAPPO Traffic Signal Control

## 1. Project Overview

`MultiJunction` is a SUMO-based reinforcement-learning project for coordinating traffic signals at four connected junctions. It extends the single-junction traffic-control work into a multi-agent setting: one agent controls each traffic light, while a shared critic learns how the combined road network is performing.

The project uses a **centralized-training, decentralized-execution** design:

- During training, a central critic receives the combined state of all four junctions.
- During testing, each traffic-light actor selects its action using only its own local traffic state.
- This architecture is commonly called **MAPPO** (Multi-Agent Proximal Policy Optimization).

## 2. Objectives

- Reduce queues and vehicle waiting time.
- Increase vehicles that complete their journeys (throughput).
- Avoid starving any approach for too long.
- Keep queues balanced across lanes.
- Avoid unnecessary rapid signal switching.
- Compare the learned controller with a fixed-time baseline under the same SUMO network and demand.

## 3. Project Structure

```text
MultiJunction/
├── algorithms/
│   ├── actor.py              # Local policy network for one traffic light
│   ├── critic.py             # Centralized value network
│   └── mappo.py              # MAPPO action selection, PPO update, save/load
├── environment/
│   └── sumo_mappo_env.py     # PettingZoo/SUMO multi-agent environment
├── models/
│   └── mappo_v2.pt           # Latest trained 10-feature MAPPO checkpoint
├── network/
│   ├── random4.net.xml       # SUMO four-junction road network
│   ├── routes.rou.xml        # Vehicle routes and departures
│   ├── trips.trips.xml       # Input trip definitions
│   └── simulation.sumocfg    # SUMO simulation configuration
├── outputs/
│   └── results/              # Timestamped MAPPO and fixed-time CSV reports
├── scripts/
│   ├── train.py              # Training and checkpoint/resume workflow
│   ├── test.py               # Detailed MAPPO or fixed-time evaluation
│   └── test_env.py           # Environment smoke test
├── requirements.txt
└── README.md
```

## 4. Requirements and Installation

### Required software

- Python 3.9 or newer.
- [Eclipse SUMO](https://sumo.dlr.de/docs/Downloads.html), with `sumo` and `sumo-gui` available on `PATH`.
- SUMO Python bindings (installed through `eclipse-sumo`).

### Install packages

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Verify that SUMO and the environment work:

```powershell
python scripts/test_env.py --steps 20
```

To see the simulation window, use:

```powershell
python scripts/test_env.py --steps 20 --gui
```

## 5. Environment and Signal Control

The environment has four agents and four SUMO traffic lights:

| Agent | Traffic-light ID |
|---|---:|
| `agent_0` | `1` |
| `agent_1` | `12` |
| `agent_2` | `2` |
| `agent_3` | `8` |

Each action is applied every **5 simulation seconds**.

| Action | Meaning |
|---:|---|
| `0` | Keep the current green phase |
| `1` | Switch between the two main green phases |

Signal operation is constrained for realism:

- Minimum green time: **10 seconds**
- Maximum green time: **45 seconds**
- A requested switch before the minimum is ignored.
- A switch is forced at the maximum green time.

## 6. State / Observation Design

Each actor receives a normalized 10-value local state vector. Values are clipped to the range 0–1.

| # | Feature | Purpose |
|---:|---|---|
| 1 | Queue length | Detect congestion at this junction |
| 2 | Total waiting time | Capture accumulated delay |
| 3 | Average speed | Measure traffic flow quality |
| 4 | Vehicle count | Represent current demand |
| 5 | Current signal phase | Tell the policy which movement has green |
| 6 | Departed vehicles | Short-term arrival/demand signal |
| 7 | Arrived vehicles | Network throughput signal |
| 8 | Queue imbalance | Encourage fair service between approaches |
| 9 | Maximum lane wait | Detect possible starvation |
| 10 | Current green duration | Help the agent decide when to switch |

The centralized critic receives the concatenation of all four local states: **4 × 10 = 40 features**.

## 7. MAPPO Implementation

### Actor networks

There are four actor networks, one per junction. Each actor is a neural network:

```text
10 inputs → Dense(128) + ReLU → Dense(128) + ReLU → 2 action probabilities
```

### Central critic

The critic estimates the value of the joint traffic state:

```text
40 inputs → Dense(128) + ReLU → Dense(128) + ReLU → 1 state-value estimate
```

### PPO learning method

After each episode, the implementation:

1. Stores local states, actions, action log probabilities, rewards, and terminal state.
2. Calculates discounted returns using `gamma = 0.99`.
3. Calculates normalized advantages using the centralized critic.
4. Updates the actors with clipped PPO ratios (`clip ratio = 0.20`).
5. Updates the critic using value loss.
6. Uses entropy regularization for exploration and gradient clipping for stable learning.

## 8. Reward Function

Each junction receives a reward with seven components:

| Component | Effect |
|---|---|
| Queue penalty | Penalizes stopped vehicles and congestion |
| Waiting-time penalty | Penalizes total lane delay |
| Queue-imbalance penalty | Encourages fair treatment of lanes |
| Throughput reward | Rewards vehicles arriving at destinations |
| Waiting-time improvement | Rewards reductions in delay since the prior decision |
| Switching penalty | Discourages unnecessary signal changes |
| Starvation penalty | Penalizes a lane waiting longer than 200 seconds |

The reward is normalized and clipped to `[-2, 2]` for stable training.

## 9. Training

### Start new training

```powershell
python scripts/train.py --episodes 500
```

### Continue an existing model

```powershell
python scripts/train.py --episodes 500 --resume
```

The training script writes `models/mappo_v2.pt`:

- Every 25 episodes by default.
- At normal completion.
- When training is stopped with `Ctrl + C`.

Useful options:

```powershell
python scripts/train.py --episodes 1000 --resume --save-every 50
python scripts/train.py --episodes 100 --gui
```

Training with GUI is useful for demonstration but slower. Use headless SUMO for long training.

## 10. Testing the Trained Model

### One headless evaluation

```powershell
python scripts/test.py --controller mappo
```

### Watch it in SUMO-GUI

```powershell
python scripts/test.py --controller mappo --gui
```

### Multiple evaluation runs

```powershell
python scripts/test.py --controller mappo --runs 3
```

Use seeded runs for reproducible comparisons. Use the same seed and number of runs for
both controllers:

```powershell
python scripts/test.py --controller fixed --green-time 30 --runs 5 --seed 100
python scripts/test.py --controller mappo --runs 5 --seed 100
```

The evaluation report includes vehicles passed, throughput, waiting time, travel time, queue length, speed, fairness, reward, CO₂, fuel, and a per-junction metrics table.

## 11. Fixed-Time Baseline and Comparison

Do not judge the learned model from its reward alone. Compare it with a non-learning fixed-time controller on the same network and demand.

Run the 30-second fixed-time baseline:

```powershell
python scripts/test.py --controller fixed --green-time 30 --runs 3
```

Run the trained MAPPO model with the same number of runs:

```powershell
python scripts/test.py --controller mappo --runs 3
```

The CSV files are saved in `outputs/results/` with timestamped names:

```text
fixed_results_YYYYMMDD_HHMMSS.csv
mappo_results_YYYYMMDD_HHMMSS.csv
```

Compare mean values across the runs:

| Metric | Better result |
|---|---|
| Vehicles passed / throughput | Higher |
| Average waiting time | Lower |
| Average travel time | Lower |
| Average queue length | Lower |
| CO₂ and fuel | Lower |
| Fairness index | Higher |

### Verified fixed-time reference run

A fixed-time 30-second run on the included network produced:

| Metric | Fixed-time result |
|---|---:|
| Vehicles passed | 442 |
| Throughput | 0.177 veh/sec |
| Average waiting time | 20.01 sec/vehicle |
| Average travel time | 80.14 sec |
| Average queue length | 23.05 vehicles |
| Average speed | 23.07 km/h |
| Evaluation reward | -194.132 |

Run MAPPO again using the current evaluation script before comparing waiting-time values, because the earlier version of the report had an incorrect system-wide waiting-time calculation.

## 12. Result Files

Each test command prints a structured terminal report and creates a CSV file under:

```text
outputs/results/
```

The CSV contains one row per run. Use it to calculate averages and create charts for a report or presentation.

## 13. Limitations and Future Work

- Results are simulation results, not proof of real-world performance.
- The included route demand is fixed; stronger evaluation should use multiple demand levels and random seeds.
- Test MAPPO and fixed-time control over several runs before making performance claims.
- A future version can add an adaptive fixed-time baseline, real detector data, emergency-vehicle priority, weather context, inter-junction communication, and travel-time-based rewards.

## 14. Suggested Presentation Statement

> This project applies Multi-Agent Proximal Policy Optimization to coordinate four SUMO traffic signals. Each junction makes local decisions from traffic conditions, while centralized training learns network-level coordination. The system is evaluated against a fixed-time controller using throughput, waiting time, queue length, travel time, emissions, fuel consumption, and fairness.
