# SingleJunction Traffic Control

This repository contains a SUMO-based single-junction traffic control experiment using reinforcement learning.

## What is included
- `scripts/main.py`: live RL inference demo using the basic PPO model.
- `scripts/fixed.py`: fixed-time traffic signal baseline with performance logging.
- `rl/basic/`: basic SUMO Gym environment and PPO training/inference.
- `rl/context/`: context-aware SUMO Gym environment and PPO training/inference.
- `models/ppo_basic/`: trained basic PPO models and normalization state.
- `models/ppo_context/`: trained context-aware PPO models and normalization state.

## Requirements
- Python 3.9+ (or a compatible Python 3 version)
- SUMO installed and available on `PATH` as `sumo` / `sumo-gui`
- `SUMO_HOME` environment variable set to the SUMO installation root
- Python packages installed from `requirements.txt`

## Install
```bash
pip install -r requirements.txt
```

If `SUMO_HOME` is not set, update it before running:
```powershell
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
```

## Run the demo
From the repository root:
```bash
python scripts/main.py
```

## Final presentation model
Use this command for the final fixed-vs-adaptive comparison. It runs the
fixed-time controller and the context-aware PPO adaptive controller with the
same simulation time and seed.

```bash
python scripts/final_demo.py --mode both --sim-time 300 --runs 3 --seed 2026 --output outputs/results/final_comparison.csv
```

Generate PPT-ready charts from the final comparison:

```bash
python scripts/final_graphs.py outputs/results/final_comparison.csv --output-dir outputs/graphs/final
```

To show the SUMO GUI during the presentation, add `--gui`:

```bash
python scripts/final_demo.py --mode both --sim-time 120 --runs 1 --seed 2026 --gui --output outputs/results/final_live_demo.csv
```

## Output folders
- `outputs/results/final_comparison.csv`: final fixed-vs-adaptive comparison table.
- `outputs/graphs/final/`: PPT-ready final comparison charts.
- `outputs/results/archive/`: older test and experiment CSV files.
- `outputs/graphs/advanced/`: earlier generated research graphs.

## Run inference
- Basic PPO model:
  ```bash
  python rl/basic/test_model.py
  ```
- Context-aware PPO model:
  ```bash
  python rl/context/test_context.py
  ```

## Run baseline
```bash
python scripts/fixed.py
```

## Training
If you need to verify training workflows, run:
```bash
python rl/basic/train.py
python rl/context/train_context.py
```

## Notes
- The scripts use `models/ppo_basic/ppo_22000.zip` and `models/ppo_context/ppo_151000.zip` by default.
- If SUMO tools are installed in a custom location, make sure `SUMO_HOME` points to the correct folder.
