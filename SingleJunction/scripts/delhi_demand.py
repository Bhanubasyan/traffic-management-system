import argparse
import csv
import json
import os
import random
import re
from xml.sax.saxutils import escape


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "Papers", "new_delhi_traffic_dataset")
WEEKDAY_DIR = os.path.join(DATASET_DIR, "weekday_stats")
GLOBAL_DIR = os.path.join(DATASET_DIR, "global_metrics")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "routes", "delhi_realistic.rou.xml")

DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

ROUTES = [
    "-E0 E1",
    "-E0 E2",
    "-E0 E3",
    "-E1 E0",
    "-E1 E2",
    "-E1 E3",
    "-E2 E0",
    "-E2 E1",
    "-E2 E3",
    "-E3 E0",
    "-E3 E1",
    "-E3 E2",
]

VEHICLE_TYPES = (
    ["car"] * 30
    + ["bike"] * 18
    + ["scooter"] * 10
    + ["auto"] * 12
    + ["jeep"] * 4
    + ["van"] * 5
    + ["public_van"] * 4
    + ["school_van"] * 3
    + ["school_bus"] * 4
    + ["truck"] * 4
    + ["oil_tanker"] * 1
    + ["water_tanker"] * 1
    + ["milk_tanker"] * 1
    + ["ambulance"] * 3
)

VTYPE_BLOCK = [
    '    <vType id="car" vClass="passenger" color="0,102,204" maxSpeed="22.2" accel="2.5"/>\n',
    '    <vType id="bike" vClass="motorcycle" color="255,69,0" maxSpeed="25" accel="3.0"/>\n',
    '    <vType id="scooter" vClass="motorcycle" color="255,140,0" maxSpeed="19.4" accel="2.5"/>\n',
    '    <vType id="auto" vClass="taxi" color="255,215,0" maxSpeed="13.8" accel="2.0"/>\n',
    '    <vType id="jeep" vClass="passenger" color="34,139,34" maxSpeed="22.2"/>\n',
    '    <vType id="van" vClass="passenger" color="169,169,169" maxSpeed="19.4"/>\n',
    '    <vType id="public_van" vClass="passenger" color="0,255,255" maxSpeed="19.4"/>\n',
    '    <vType id="school_van" vClass="passenger" color="255,165,0" maxSpeed="16.6"/>\n',
    '    <vType id="school_bus" vClass="bus" color="255,223,0" maxSpeed="16.6"/>\n',
    '    <vType id="truck" vClass="truck" color="139,69,19" maxSpeed="16.6"/>\n',
    '    <vType id="oil_tanker" vClass="truck" color="0,0,0" maxSpeed="15"/>\n',
    '    <vType id="water_tanker" vClass="truck" color="30,144,255" maxSpeed="15"/>\n',
    '    <vType id="milk_tanker" vClass="truck" color="255,255,255" maxSpeed="15"/>\n',
    '    <vType id="ambulance" vClass="emergency" color="255,0,0" maxSpeed="27.7" accel="3.5"/>\n',
]

PROFILE_DEFAULTS = {
    "offpeak": ("Sunday", "04:00 AM"),
    "morning": ("Monday", "08:00 AM"),
    "midday": ("Wednesday", "12:00 PM"),
    "evening": ("Friday", "06:00 PM"),
    "festival": ("Monday", "06:00 PM"),
}


def parse_duration_seconds(value):
    total = 0
    match = re.search(r"(\d+)\s*min", value)
    if match:
        total += int(match.group(1)) * 60
    match = re.search(r"(\d+)\s*s", value)
    if match:
        total += int(match.group(1))
    return total


def parse_speed(value):
    match = re.search(r"([\d.]+)", value)
    return float(match.group(1)) if match else 0.0


def parse_percent(value):
    match = re.search(r"([\d.]+)", value)
    return float(match.group(1)) if match else 0.0


def read_metric_csv(filename, parser):
    path = os.path.join(WEEKDAY_DIR, filename)
    with open(path, newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        return {
            row["Time"]: {
                day: parser(row[day])
                for day in DAYS
            }
            for row in reader
        }


def read_delhi_profiles():
    time_by_day = read_metric_csv("2024_week_day_time_city.csv", parse_duration_seconds)
    speed_by_day = read_metric_csv("2024_week_day_speed_city.csv", parse_speed)
    congestion_by_day = read_metric_csv("2024_week_day_congestion_city.csv", parse_percent)

    rush_path = os.path.join(GLOBAL_DIR, "2024_city_rush_hour.json")
    with open(rush_path, encoding="utf-8") as file_obj:
        rush_metrics = json.load(file_obj)

    return time_by_day, speed_by_day, congestion_by_day, rush_metrics


def profile_context(profile, day, time_label):
    time_by_day, speed_by_day, congestion_by_day, rush_metrics = read_delhi_profiles()

    if profile == "morning":
        rush = rush_metrics["morning_rush_hour"]
        return {
            "travel_time_10km_sec": parse_duration_seconds(rush["time_taken_10km"]),
            "speed_kmph": float(rush["average_speed_kmh"]),
            "congestion_percent": float(rush["congestion_level_percent"]),
            "rush_type": "morning_rush",
        }

    if profile in ("evening", "festival"):
        rush = rush_metrics["evening_rush_hour"]
        congestion = float(rush["congestion_level_percent"])
        if profile == "festival":
            congestion = min(85.0, congestion + 10.0)
        return {
            "travel_time_10km_sec": parse_duration_seconds(rush["time_taken_10km"]),
            "speed_kmph": float(rush["average_speed_kmh"]),
            "congestion_percent": congestion,
            "rush_type": "evening_rush" if profile == "evening" else "festival_peak",
        }

    return {
        "travel_time_10km_sec": time_by_day[time_label][day],
        "speed_kmph": speed_by_day[time_label][day],
        "congestion_percent": congestion_by_day[time_label][day],
        "rush_type": profile,
    }


def vehicle_id(vehicle_type, count):
    if vehicle_type == "bike":
        return f"D{count}"
    if "tanker" in vehicle_type:
        return f"DTK{count}"
    prefixes = {
        "car": "DC",
        "truck": "DT",
        "ambulance": "DA",
        "school_bus": "DB",
        "auto": "DR",
        "scooter": "DS",
    }
    return f"{prefixes.get(vehicle_type, 'DV')}{count}"


def target_vehicle_count(sim_time, congestion_percent, scale):
    vehicles_per_600 = 160 + (congestion_percent * 5.5)
    return max(40, int(round((vehicles_per_600 * sim_time / 600.0) * scale)))


def weighted_route(rng, congestion_percent):
    # Put more demand on two opposing approaches during high congestion. This
    # creates directional imbalance like real rush-hour flows.
    if congestion_percent >= 50:
        routes = ["-E0 E2", "-E2 E0", "-E1 E3", "-E3 E1"] + ROUTES
        weights = [8, 8, 5, 5] + [2] * len(ROUTES)
        return rng.choices(routes, weights=weights, k=1)[0]
    return rng.choice(ROUTES)


def weighted_vehicle_type(rng, profile):
    vehicle_types = list(VEHICLE_TYPES)
    if profile == "morning":
        vehicle_types += ["school_bus"] * 4 + ["school_van"] * 5 + ["public_van"] * 3
    elif profile in ("evening", "festival"):
        vehicle_types += ["car"] * 8 + ["auto"] * 5 + ["bike"] * 5 + ["truck"] * 2
    return rng.choice(vehicle_types)


def generate_depart_times(rng, vehicle_count, sim_time, congestion_percent):
    times = []
    for _ in range(vehicle_count):
        if congestion_percent >= 50 and rng.random() < 0.65:
            depart = rng.gauss(sim_time * 0.58, sim_time * 0.18)
        elif congestion_percent >= 25 and rng.random() < 0.45:
            depart = rng.gauss(sim_time * 0.50, sim_time * 0.24)
        else:
            depart = rng.uniform(0, sim_time)
        times.append(max(0.0, min(sim_time - 1, depart)))
    return sorted(times)


def write_route_file(output, profile, day, time_label, sim_time, seed, scale):
    rng = random.Random(seed)
    context = profile_context(profile, day, time_label)
    congestion = context["congestion_percent"]
    vehicle_count = target_vehicle_count(sim_time, congestion, scale)
    depart_times = generate_depart_times(rng, vehicle_count, sim_time, congestion)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as file_obj:
        file_obj.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file_obj.write("<!-- Generated from New Delhi 2024 traffic analytics. -->\n")
        file_obj.write(
            f"<!-- profile={profile}, day={day}, time={time_label}, "
            f"congestion={congestion:.1f}%, speed={context['speed_kmph']:.1f} km/h, "
            f"travel_time_10km={context['travel_time_10km_sec']} sec, seed={seed} -->\n"
        )
        file_obj.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
        file_obj.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        for line in VTYPE_BLOCK:
            file_obj.write(line)

        for index, depart in enumerate(depart_times):
            vehicle_type = weighted_vehicle_type(rng, profile)
            route_edges = weighted_route(rng, congestion)
            file_obj.write(
                f'    <vehicle id="{escape(vehicle_id(vehicle_type, index))}" '
                f'depart="{depart:.2f}" type="{escape(vehicle_type)}">\n'
            )
            file_obj.write(f'        <route edges="{escape(route_edges)}"/>\n')
            file_obj.write("    </vehicle>\n")

        file_obj.write("</routes>\n")

    return vehicle_count, context


def main():
    parser = argparse.ArgumentParser(
        description="Generate a SUMO route file from New Delhi traffic analytics."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default="evening",
        help="Delhi demand profile to generate.",
    )
    parser.add_argument("--day", choices=DAYS, help="Override weekday from the selected profile.")
    parser.add_argument("--time", help='Override time label, for example "08:00 AM".')
    parser.add_argument("--sim-time", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    default_day, default_time = PROFILE_DEFAULTS[args.profile]
    day = args.day or default_day
    time_label = args.time or default_time

    vehicle_count, context = write_route_file(
        args.output,
        args.profile,
        day,
        time_label,
        args.sim_time,
        args.seed,
        args.scale,
    )

    print("Delhi demand route file generated")
    print(f"Output                 : {args.output}")
    print(f"Profile                : {args.profile}")
    print(f"Day / time             : {day} / {time_label}")
    print(f"Simulation time        : {args.sim_time} sec")
    print(f"Vehicles generated     : {vehicle_count}")
    print(f"Delhi congestion       : {context['congestion_percent']:.1f}%")
    print(f"Delhi average speed    : {context['speed_kmph']:.1f} km/h")
    print(f"Delhi travel time 10km : {context['travel_time_10km_sec']} sec")


if __name__ == "__main__":
    main()
