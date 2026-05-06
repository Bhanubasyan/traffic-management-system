import traci
import time

sumoCmd = ["sumo-gui", "-c", "config/simulation.sumocfg"]
traci.start(sumoCmd)

print("🔥 Heatmap Mode Running")

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

    for edge in traci.edge.getIDList():

        if edge.startswith(":"):
            continue

        count = traci.edge.getLastStepVehicleNumber(edge)

        if count > 20:
            traci.edge.setParameter(edge, "color", "255,0,0")     # RED
        elif count > 8:
            traci.edge.setParameter(edge, "color", "255,165,0")   # ORANGE
        else:
            traci.edge.setParameter(edge, "color", "0,255,0")     # GREEN

    time.sleep(0.2)

traci.close()