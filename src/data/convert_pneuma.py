import os
import csv
from glob import glob

MAX_VEHICLES_PER_FILE = 100

csv_files = glob("pNEUMA_dataset/*.csv")

os.makedirs("processed", exist_ok=True)

output_file = "processed/cleaned_data.csv"

with open(output_file, "w", newline="", encoding="utf-8") as out:

    writer = csv.writer(out)

    writer.writerow([
        "track_id",
        "type",
        "traveled_d",
        "avg_speed",
        "lat",
        "lon",
        "speed",
        "lon_acc",
        "lat_acc",
        "time"
    ])

    for idx, file in enumerate(csv_files, start=1):

        print(f"[{idx}/{len(csv_files)}] Processing {os.path.basename(file)}")

        with open(file, "r", encoding="utf-8") as f:

            # bỏ header
            next(f)

            for line_num, line in enumerate(f, start=1):

                if line_num > MAX_VEHICLES_PER_FILE:
                    break

                values = line.strip().split(";")

                if len(values) < 10:
                    continue

                track_id = values[0].strip()
                vehicle_type = values[1].strip()
                traveled_d = values[2].strip()
                avg_speed = values[3].strip()

                trajectory = values[4:]

                MAX_POINTS_PER_VEHICLE = 100

                for i in range(
                    0,
                    min(len(trajectory), MAX_POINTS_PER_VEHICLE * 6),
                    6
                ):

                    if i + 5 >= len(trajectory):
                        break

                    writer.writerow([
                        track_id,
                        vehicle_type,
                        traveled_d,
                        avg_speed,
                        trajectory[i],
                        trajectory[i + 1],
                        trajectory[i + 2],
                        trajectory[i + 3],
                        trajectory[i + 4],
                        trajectory[i + 5]
                    ])

                if line_num % 1000 == 0:
                    print(
                        f"   processed {line_num:,} vehicles"
                    )

print(f"Saved to: {output_file}")