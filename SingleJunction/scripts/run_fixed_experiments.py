import subprocess
import time

NUM_RUNS = 30

print("\n========== STARTING FIXED SYSTEM EXPERIMENTS ==========\n")

for i in range(NUM_RUNS):

    print(f"\n🚀 Running Simulation {i+1}/{NUM_RUNS}\n")

    subprocess.run(["python", "scripts/main.py"])

    time.sleep(2)

print("\n========== ALL FIXED EXPERIMENTS COMPLETED ==========\n")