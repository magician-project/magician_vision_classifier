import numpy as np
import csv

CSV_PATH = "camera_poses.csv"

NUM_CAMERAS = 10
RADIUS = 5.0    # distance from origin
HEIGHT = 1.5    # camera height

rows = []

for i in range(NUM_CAMERAS):
    angle = 2.0 * np.pi * i / NUM_CAMERAS
    x = RADIUS * np.cos(angle)
    z = RADIUS * np.sin(angle)
    y = HEIGHT

    # Point toward origin in XZ plane:
    # yaw around Y axis: atan2(-z, -x) (to face (0,0,0))
    yaw = np.degrees(np.arctan2(-z, -x))

    # Simple pitch: look slightly downward
    pitch = -10.0  # degrees
    roll = 0.0

    rows.append([x, y, z, pitch, yaw, roll])

# Write CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    for r in rows:
        writer.writerow(r)

print("Saved camera poses to:", CSV_PATH)

