import random

# 🔹 file paths
input_file = "routes/updated.rou.xml"
output_file = "routes/updated.rou.xml"

with open(input_file, "r") as f:
    data = f.readlines()

# 🔥 realistic distribution
types = (
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

new_data = []
count = 0

for line in data:
    if "<vehicle" in line:

        t = random.choice(types)

        # 🔹 LABEL SYSTEM
        if t == "bike":
            new_id = f"{count}"   # no label
        elif t == "car":
            new_id = f"C{count}"
        elif t == "truck":
            new_id = f"T{count}"
        elif t == "ambulance":
            new_id = f"A{count}"
        elif t == "bus" or t == "school_bus":
            new_id = f"B{count}"
        elif t == "auto":
            new_id = f"R{count}"  # rickshaw
        elif t == "scooter":
            new_id = f"S{count}"
        elif "tanker" in t:
            new_id = f"TK{count}"
        else:
            new_id = f"V{count}"

        line = line.replace('id="', f'id="{new_id}')
        line = line.replace(">", f' type="{t}">')

        count += 1

    new_data.append(line)

# 🔹 insert vType AFTER <routes>
insert_index = 0
for i, line in enumerate(new_data):
    if "<routes" in line:
        insert_index = i + 1
        break

# 🎨 BETTER COLORS (distinct + realistic)

    # 🎯 REALISTIC SPEEDS (m/s)
vtype_block = [
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

    '    <vType id="ambulance" vClass="emergency" color="255,0,0" maxSpeed="27.7" accel="3.5"/>\n'
]
new_data = new_data[:insert_index] + vtype_block + new_data[insert_index:]

# 🔹 save file
with open(output_file, "w") as f:
    f.writelines(new_data)

print("✅ FINAL FILE READY:", output_file)