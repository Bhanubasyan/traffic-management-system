import traci
import os

# =========================================================
#               FIXED TRAFFIC CONTROL MODEL
# =========================================================
# Platform  : SUMO + TraCI
# Type      : Fixed Traffic Signal System
# Features  :
#   ✅ Fixed cyclic signal control
#   ✅ Vehicle waiting time
#   ✅ Travel time calculation
#   ✅ Fuel consumption estimation
#   ✅ CO2 emission monitoring
#   ✅ Traffic throughput
#   ✅ Lane-wise vehicle count
# =========================================================


# ================= START SUMO =================
sumoCmd = ["sumo-gui", "-c", "config/simulation.sumocfg"]
traci.start(sumoCmd)

# ================= TRAFFIC LIGHT IDS =================
tls_ids = traci.trafficlight.getIDList()

# ================= SIMULATION PARAMETERS =================
SIM_TIME = 300

# ================= FIXED SIGNAL TIMING =================
GREEN_TIME = 20
YELLOW_TIME = 5

# ================= VEHICLE CROSSING TIMES =================
# average crossing time (seconds)
vehicle_cross_time = {
    "car": 2,
    "bus": 4,
    "truck": 5,
    "motorcycle": 1,
    "bike": 1,
    "rickshaw": 3
}

# ================= FUEL CONSUMPTION RATE =================
# ml per second
fuel_rate = {
    "car": 0.0002,
    "bus": 0.0006,
    "truck": 0.0008,
    "motorcycle": 0.0001,
    "bike": 0.00005,
    "rickshaw": 0.00015
}

# ================= PERFORMANCE TRACKERS =================
vehicle_entry_time = {}
vehicle_exit_time = {}
vehicle_last_lane = {}

lane_pass_count = {}

total_vehicles_passed = 0
total_waiting_time = 0
vehicle_waiting = {}
total_travel_time = 0

fuel_consumption = 0

emission_history = []

# ================= PHASE TRACKING =================
current_phase_index = {}

for tl in tls_ids:
    current_phase_index[tl] = 0

# =========================================================
#               GREEN TIME ESTIMATION
# =========================================================
def estimate_green_time(lanes):

    nc = 0
    nr = 0
    nb = 0
    nt = 0
    nbike = 0

    for lane in lanes:

        vehicles = traci.lane.getLastStepVehicleIDs(lane)

        for v in vehicles:

            try:
                vtype = traci.vehicle.getTypeID(v).lower()

                if "car" in vtype:
                    nc += 1

                elif "rickshaw" in vtype:
                    nr += 1

                elif "bus" in vtype:
                    nb += 1

                elif "truck" in vtype:
                    nt += 1

                else:
                    nbike += 1

            except:
                pass

    L = max(len(lanes), 1)

    numerator = (
        (nc * vehicle_cross_time["car"]) +
        (nr * vehicle_cross_time["rickshaw"]) +
        (nb * vehicle_cross_time["bus"]) +
        (nt * vehicle_cross_time["truck"]) +
        (nbike * vehicle_cross_time["bike"])
    )

    Tgreen = numerator / (L + 1)

    # bounded green time
    Tgreen = max(10, min(Tgreen, 60))

    return int(Tgreen)

# =========================================================
#                       MAIN LOOP
# =========================================================
while traci.simulation.getMinExpectedNumber() > 0:

    current_time = traci.simulation.getTime()

    if current_time >= SIM_TIME:
        break

    traci.simulationStep()

    # =====================================================
    #               VEHICLE TRACKING
    # =====================================================
    vehicles = traci.vehicle.getIDList()

    total_emission = 0

    for v in vehicles:

        # entry time
        if v not in vehicle_entry_time:
            vehicle_entry_time[v] = current_time

        # waiting time
        waiting = traci.vehicle.getAccumulatedWaitingTime(v)

        vehicle_waiting[v] = max(
        vehicle_waiting.get(v, 0),
        waiting
        )

       
        # current lane
        try:
            vehicle_last_lane[v] = traci.vehicle.getLaneID(v)
        except:
            pass

        # =================================================
        #               FUEL CONSUMPTION
        # =================================================
        try:

            vtype = traci.vehicle.getTypeID(v).lower()

            rate = 0.0002

            if "car" in vtype:
                rate = fuel_rate["car"]

            elif "bus" in vtype:
                rate = fuel_rate["bus"]

            elif "truck" in vtype:
                rate = fuel_rate["truck"]

            elif "rickshaw" in vtype:
                rate = fuel_rate["rickshaw"]

            elif "motorcycle" in vtype:
                rate = fuel_rate["motorcycle"]

            else:
                rate = fuel_rate["bike"]

            fuel_consumption += waiting * rate

        except:
            pass

        # =================================================
        #               CO2 EMISSION
        # =================================================
        try:
            total_emission += traci.vehicle.getCO2Emission(v)
        except:
            pass

    emission_history.append(total_emission)

    # =====================================================
    #               FIXED SIGNAL CONTROL
    # =====================================================
    for tl in tls_ids:

        logic = traci.trafficlight.getAllProgramLogics(tl)[0]

        total_phases = len(logic.phases)

        # fixed cycle
        cycle_time = GREEN_TIME + YELLOW_TIME

        if current_time % cycle_time == 0:

            current_phase = current_phase_index[tl]

            next_phase = (current_phase + 1) % total_phases

            # =================================================
            # OPTIONAL GREEN ESTIMATION
            # =================================================
            lanes = traci.trafficlight.getControlledLanes(tl)

            estimated_green = estimate_green_time(lanes)

            final_green = GREEN_TIME

            # =================================================
            # APPLY SIGNAL
            # =================================================
            traci.trafficlight.setPhase(tl, next_phase)

            traci.trafficlight.setPhaseDuration(
                tl,
                final_green
            )

            current_phase_index[tl] = next_phase

            print(
                f"🚦 TL: {tl} | "
                f"Phase: {next_phase} | "
                f"Green Time: {final_green}"
            )

    # =====================================================
    #               EXIT TRACKING
    # =====================================================
    current_vehicles = set(traci.vehicle.getIDList())

    for v in list(vehicle_entry_time.keys()):

        if v not in current_vehicles and v not in vehicle_exit_time:

            exit_t = current_time

            vehicle_exit_time[v] = exit_t

            # travel time
            travel_time = exit_t - vehicle_entry_time[v]

            total_travel_time += travel_time

            total_vehicles_passed += 1

            # lane count
            lane = vehicle_last_lane.get(v, "unknown")

            lane_pass_count[lane] = (
                lane_pass_count.get(lane, 0) + 1
            )

# =========================================================
#                   FINAL RESULT
# =========================================================
print("\n========== 🚦 FIXED SIGNAL RESULT ==========")

print(f"\nSimulation Time: {SIM_TIME} sec")

print(f"\nTotal Vehicles Passed: {total_vehicles_passed}")

# =========================================================
#               AVERAGE WAITING TIME
# =========================================================
total_waiting_time = sum(vehicle_waiting.values())
avg_wait = (
    total_waiting_time / len(vehicle_waiting)
    if len(vehicle_waiting) > 0 else 0
)

print(f"\n⏳ Average Waiting Time: {avg_wait:.2f}")

# =========================================================
#               AVERAGE TRAVEL TIME
# =========================================================
avg_travel = (
    total_travel_time / total_vehicles_passed
    if total_vehicles_passed > 0 else 0
)

print(f"⏱ Average Travel Time: {avg_travel:.2f}")

# =========================================================
#               FUEL CONSUMPTION
# =========================================================
print(f"\n⛽ Total Fuel Consumption: {fuel_consumption:.6f} ml")

# =========================================================
#               CO2 EMISSION
# =========================================================
total_co2 = sum(emission_history)

avg_co2 = (
    total_co2 / len(emission_history)
    if len(emission_history) > 0 else 0
)

print(f"\n🌿 Total CO2 Emission: {total_co2:.2f}")

print(f"🌿 Average CO2 per Step: {avg_co2:.2f}")

# =========================================================
#               TRAFFIC THROUGHPUT
# =========================================================
throughput = total_vehicles_passed / SIM_TIME

print(f"\n🚗 Traffic Throughput: {throughput:.2f} vehicles/sec")

# =========================================================
#               LANE-WISE COUNT
# =========================================================
print("\n📍 Lane-wise Vehicle Count:")

for lane, count in lane_pass_count.items():

    print(f"{lane} → {count}")

print("\n============================================")

# ================= CLOSE =================
traci.close()