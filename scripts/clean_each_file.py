import os
import csv
from glob import glob

input_folder = "pNEUMA_dataset"
output_folder = "processed"

os.makedirs(output_folder, exist_ok=True)

csv_files = glob(os.path.join(input_folder, "*.csv"))

for file_idx, file in enumerate(csv_files, start=1):

    filename = os.path.basename(file)
    output_file = os.path.join(
        output_folder,
        f"clean_{filename}"
    )

    print(f"\n[{file_idx}/{len(csv_files)}] {filename}")

    saved_rows = 0
    skipped_rows = 0

    with open(file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)

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

        next(fin)

        for line in fin:

            values = line.strip().split(";")

            if len(values) < 10:
                skipped_rows += 1
                continue

            try:
                track_id = int(values[0])
                vehicle_type = values[1]

                traveled_d = float(values[2])
                avg_speed = float(values[3])

            except ValueError:
                skipped_rows += 1
                continue

            if traveled_d < 0 or avg_speed < 0:
                skipped_rows += 1
                continue

            trajectory = values[4:]

            for i in range(0, len(trajectory), 6):

                if i + 5 >= len(trajectory):
                    break

                try:
                    lat = float(trajectory[i])
                    lon = float(trajectory[i + 1])
                    speed = float(trajectory[i + 2])
                    lon_acc = float(trajectory[i + 3])
                    lat_acc = float(trajectory[i + 4])
                    time = float(trajectory[i + 5])

                except ValueError:
                    continue

                # GPS validation
                if not (-90 <= lat <= 90):
                    continue

                if not (-180 <= lon <= 180):
                    continue

                # abnormal values
                if speed < 0:
                    continue

                if time < 0:
                    continue

                writer.writerow([
                    track_id,
                    vehicle_type,
                    traveled_d,
                    avg_speed,
                    lat,
                    lon,
                    speed,
                    lon_acc,
                    lat_acc,
                    time
                ])

                saved_rows += 1

    print(f"Saved : {saved_rows:,}")
    print(f"Skipped : {skipped_rows:,}")
    print(f"Output : {output_file}")