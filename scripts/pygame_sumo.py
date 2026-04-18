import traci
import pygame
import sys

# 🔹 SUMO start (IMPORTANT: yahi tera config use karega)
sumoCmd = ["sumo-gui", "-c", "config/simulation.sumocfg"]
traci.start(sumoCmd)

# 🔹 PyGame init
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🚦 SUMO + PyGame Traffic System")

clock = pygame.time.Clock()

# 🔹 coordinate convert (SUMO → screen)
def convert(x, y):
    scale = 2   # zoom control
    return int(x * scale), int(HEIGHT - y * scale)

# 🔹 main loop
running = True

while running:
    clock.tick(30)

    # window close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # SUMO simulation step
    traci.simulationStep()

    screen.fill((25, 25, 25))  # dark background

    vehicles = traci.vehicle.getIDList()

    for v in vehicles:
        x, y = traci.vehicle.getPosition(v)
        x, y = convert(x, y)

        # 🔹 color logic
        if v.startswith("C"):
            color = (0, 102, 204)   # car blue
        elif v.startswith("T"):
            color = (139, 69, 19)   # truck brown
        elif v.startswith("A"):
            color = (255, 0, 0)     # ambulance red
        else:
            color = (0, 255, 0)

        pygame.draw.rect(screen, color, (x, y, 12, 6))

    pygame.display.flip()

# 🔹 close
traci.close()
pygame.quit()
sys.exit()