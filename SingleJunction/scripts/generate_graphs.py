import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= LOAD DATA =================

fixed_df = pd.read_csv("fixed_results.csv")
rl_df = pd.read_csv("rl_results.csv")

# ================= OUTPUT FOLDER =================

output_dir = "advanced_graphs"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
#               1. BOXPLOT COMPARISON
# =========================================================

metrics = [
    ("WaitingTime", "Waiting Time (sec)"),
    ("TravelTime", "Travel Time (sec)"),
    ("Throughput", "Throughput (veh/sec)"),
    ("FuelConsumption", "Fuel Consumption (L)"),
    ("CO2Emission", "CO2 Emission (kg)")
]

for metric, ylabel in metrics:

    plt.figure(figsize=(7, 5))

    data = [
        fixed_df[metric],
        rl_df[metric]
    ]

    plt.boxplot(
        data,
        labels=["Fixed", "Adaptive RL"],
        patch_artist=True
    )

    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Distribution Comparison")

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(
        os.path.join(output_dir, f"boxplot_{metric}.png"),
        bbox_inches="tight"
    )

    plt.close()

# =========================================================
#               2. RUN-WISE TREND GRAPHS
# =========================================================

for metric, ylabel in metrics:

    plt.figure(figsize=(8, 5))

    plt.plot(
        fixed_df.index + 1,
        fixed_df[metric],
        marker='o',
        label="Fixed"
    )

    plt.plot(
        rl_df.index + 1,
        rl_df[metric],
        marker='s',
        label="Adaptive RL"
    )

    plt.xlabel("Simulation Run")
    plt.ylabel(ylabel)

    plt.title(f"{ylabel} Across Simulation Runs")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(
        os.path.join(output_dir, f"trend_{metric}.png"),
        bbox_inches="tight"
    )

    plt.close()

# =========================================================
#               3. RADAR CHART
# =========================================================

radar_metrics = [
    "WaitingTime",
    "TravelTime",
    "FuelConsumption",
    "CO2Emission"
]

fixed_values = [
    fixed_df[m].mean()
    for m in radar_metrics
]

rl_values = [
    rl_df[m].mean()
    for m in radar_metrics
]

# Normalize
max_vals = np.maximum(fixed_values, rl_values)

fixed_norm = np.array(fixed_values) / max_vals
rl_norm = np.array(rl_values) / max_vals

labels = [
    "Waiting",
    "Travel",
    "Fuel",
    "CO2"
]

angles = np.linspace(
    0,
    2 * np.pi,
    len(labels),
    endpoint=False
).tolist()

fixed_norm = np.concatenate((fixed_norm, [fixed_norm[0]]))
rl_norm = np.concatenate((rl_norm, [rl_norm[0]]))
angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(7, 7),
    subplot_kw=dict(polar=True)
)

ax.plot(angles, fixed_norm, linewidth=2, label="Fixed")
ax.fill(angles, fixed_norm, alpha=0.25)

ax.plot(angles, rl_norm, linewidth=2, label="Adaptive RL")
ax.fill(angles, rl_norm, alpha=0.25)

ax.set_thetagrids(
    np.degrees(angles[:-1]),
    labels
)

plt.title("Overall System Performance Comparison")

plt.legend(loc='upper right')

plt.savefig(
    os.path.join(output_dir, "radar_comparison.png"),
    bbox_inches="tight"
)

plt.close()

# =========================================================
#               4. VEHICLES PASSED COMPARISON
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    fixed_df.index + 1,
    fixed_df["VehiclesPassed"],
    marker='o',
    label="Fixed"
)

plt.plot(
    rl_df.index + 1,
    rl_df["VehiclesPassed"],
    marker='s',
    label="Adaptive RL"
)

plt.xlabel("Simulation Run")
plt.ylabel("Vehicles Passed")

plt.title("Vehicles Passed Across Simulation Runs")

plt.legend()

plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig(
    os.path.join(output_dir, "vehicles_passed_trend.png"),
    bbox_inches="tight"
)

plt.close()

# =========================================================
#               5. SUMMARY STATISTICS CSV
# =========================================================

summary = pd.DataFrame()

for metric, _ in metrics:

    summary.loc[metric, "Fixed Mean"] = fixed_df[metric].mean()
    summary.loc[metric, "Fixed Std"] = fixed_df[metric].std()

    summary.loc[metric, "RL Mean"] = rl_df[metric].mean()
    summary.loc[metric, "RL Std"] = rl_df[metric].std()

summary.to_csv(
    os.path.join(output_dir, "summary_statistics.csv")
)

print("\n✅ ADVANCED RESEARCH GRAPHS GENERATED")
print(f"📁 Saved in folder: {output_dir}")