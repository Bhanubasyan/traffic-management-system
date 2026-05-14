import argparse
import random
import re


# Realistic vehicle distribution.
VEHICLE_TYPES = (
    ["car"] * 25 +
    ["bike"] * 20 +
    ["scooter"] * 10 +
    ["auto"] * 10 +
    ["jeep"] * 5 +
    ["van"] * 5 +
    ["public_van"] * 5 +
    ["school_van"] * 5 +
    ["school_bus"] * 5 +
    ["truck"] * 3 +
    ["oil_tanker"] * 2 +
    ["water_tanker"] * 2 +
    ["milk_tanker"] * 2 +
    ["ambulance"] * 5
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

ID_PREFIX = {
    "car": "C",
    "truck": "T",
    "ambulance": "A",
    "school_bus": "B",
    "auto": "R",
    "scooter": "S",
}


def vehicle_id(vehicle_type, count):
    if vehicle_type == "bike":
        return str(count)
    if "tanker" in vehicle_type:
        return f"TK{count}"
    return f"{ID_PREFIX.get(vehicle_type, 'V')}{count}"


def update_vehicle_types(input_file="routes/routes.rou.xml", output_file=None, seed=None):
    output_file = output_file or input_file
    rng = random.Random(seed)

    with open(input_file, "r", encoding="utf-8") as f:
        data = f.readlines()

    new_data = []
    count = 0

    for line in data:
        if re.search(r"<vType\s", line):
            continue

        if re.search(r"<vehicle\s", line):
            selected_type = rng.choice(VEHICLE_TYPES)
            new_id = vehicle_id(selected_type, count)

            line = re.sub(r'id="[^"]*"', f'id="{new_id}"', line, count=1)
            line = re.sub(r'\s+type="[^"]*"', "", line, count=1)
            line = line.replace(">", f' type="{selected_type}">', 1)

            count += 1

        new_data.append(line)

    insert_index = 0
    for i, line in enumerate(new_data):
        if "<routes" in line:
            insert_index = i + 1
            break

    new_data = new_data[:insert_index] + VTYPE_BLOCK + new_data[insert_index:]

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(new_data)

    print(f"Vehicle types added to {output_file} ({count} vehicles)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="routes/routes.rou.xml")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    update_vehicle_types(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
