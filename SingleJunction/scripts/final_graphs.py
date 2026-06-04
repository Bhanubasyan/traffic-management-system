import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("WaitingTime", "Average Waiting Time", "sec/vehicle", False),
    ("TravelTime", "Average Travel Time", "sec", False),
    ("Throughput", "Throughput", "vehicles/sec", True),
    ("VehiclesPassed", "Vehicles Passed", "vehicles", True),
    ("PassedCars", "Passed Cars", "vehicles", True),
    ("PassedBikes", "Passed Bikes", "vehicles", True),
    ("PassedTrucks", "Passed Trucks", "vehicles", True),
    ("PassedBuses", "Passed Buses", "vehicles", True),
    ("PassedMotorcycles", "Passed Motorcycles", "vehicles", True),
    ("PassedRickshaws", "Passed Rickshaws", "vehicles", True),
    ("PassedEmergencyVehicles", "Passed Emergency Vehicles", "vehicles", True),
    ("FuelConsumption", "Fuel Consumption", "L", False),
    ("CO2Emission", "CO2 Emission", "kg", False),
    ("AvgQueue", "Average Queue Length", "vehicles", False),
    ("AvgSpeedKmph", "Average Speed", "km/h", True),
    ("FairnessIndex", "Fairness Index", "score", True),
]


def improvement_text(fixed, adaptive, higher_is_better):
    if fixed == 0:
        return "n/a"
    if higher_is_better:
        change = ((adaptive - fixed) / fixed) * 100
    else:
        change = ((fixed - adaptive) / fixed) * 100
    return f"{change:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Generate PPT-ready graphs from final_demo.py output.")
    parser.add_argument("csv_file", help="CSV file produced by scripts/final_demo.py")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "graphs", "final"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    os.makedirs(args.output_dir, exist_ok=True)

    summary = df.groupby("System", as_index=True).mean(numeric_only=True)
    summary.to_csv(os.path.join(args.output_dir, "final_summary_statistics.csv"))

    for metric, title, unit, higher_is_better in METRICS:
        if metric not in df.columns:
            continue

        means = summary[metric].dropna()
        systems = means.index.tolist()
        values = means.values

        plt.figure(figsize=(7, 4.5))
        colors = ["#4c78a8" if "Fixed" in system else "#59a14f" for system in systems]
        bars = plt.bar(systems, values, color=colors)
        plt.ylabel(unit)
        plt.title(f"{title}: Fixed vs Adaptive")
        plt.grid(axis="y", linestyle="--", alpha=0.35)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        if "Fixed" in means.index and "Adaptive Context PPO" in means.index:
            fixed = means["Fixed"]
            adaptive = means["Adaptive Context PPO"]
            text = improvement_text(fixed, adaptive, higher_is_better)
            plt.figtext(
                0.5,
                0.01,
                f"Adaptive improvement: {text}",
                ha="center",
                fontsize=10,
            )

        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.savefig(os.path.join(args.output_dir, f"{metric}_comparison.png"), dpi=180)
        plt.close()

    radar_metrics = ["WaitingTime", "TravelTime", "FuelConsumption", "CO2Emission", "AvgQueue"]
    if all(metric in summary.columns for metric in radar_metrics):
        fixed_values = summary.loc["Fixed", radar_metrics].to_numpy(dtype=float)
        adaptive_values = summary.loc["Adaptive Context PPO", radar_metrics].to_numpy(dtype=float)
        max_values = np.maximum(fixed_values, adaptive_values)
        max_values[max_values == 0] = 1

        fixed_norm = fixed_values / max_values
        adaptive_norm = adaptive_values / max_values
        labels = ["Waiting", "Travel", "Fuel", "CO2", "Queue"]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        fixed_norm = np.concatenate([fixed_norm, [fixed_norm[0]]])
        adaptive_norm = np.concatenate([adaptive_norm, [adaptive_norm[0]]])
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.plot(angles, fixed_norm, linewidth=2, label="Fixed", color="#4c78a8")
        ax.fill(angles, fixed_norm, alpha=0.22, color="#4c78a8")
        ax.plot(angles, adaptive_norm, linewidth=2, label="Adaptive Context PPO", color="#59a14f")
        ax.fill(angles, adaptive_norm, alpha=0.22, color="#59a14f")
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("Overall Cost Metrics")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "overall_radar_comparison.png"), dpi=180)
        plt.close()

    type_metrics = [
        "PassedCars",
        "PassedBikes",
        "PassedTrucks",
        "PassedBuses",
        "PassedMotorcycles",
        "PassedRickshaws",
        "PassedEmergencyVehicles",
    ]
    if all(metric in summary.columns for metric in type_metrics):
        labels = ["Cars", "Bikes", "Trucks", "Buses", "Motorcycles", "Rickshaws", "Emergency"]
        systems = summary.index.tolist()
        bottoms = np.zeros(len(systems))
        colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2", "#ff9da6"]

        plt.figure(figsize=(8, 5))
        for metric, label, color in zip(type_metrics, labels, colors):
            values = summary[metric].to_numpy(dtype=float)
            plt.bar(systems, values, bottom=bottoms, label=label, color=color)
            bottoms += values

        plt.ylabel("vehicles")
        plt.title("Passed Vehicle Type Breakdown")
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "vehicle_type_breakdown.png"), dpi=180)
        plt.close()

    print(f"Saved final PPT graphs to {args.output_dir}")


if __name__ == "__main__":
    main()
