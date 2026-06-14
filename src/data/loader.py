from pathlib import Path
import pandas as pd

def load_pneuma(filepath):

    records = []

    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf8") as f:

        for line in f:

            parts = [x.strip() for x in line.strip().split(";")]

            if len(parts) < 10:
                continue

            if parts[0].lower() == "track_id":
                continue

            track_id = parts[0]
            type = parts[1]

            try:
                traveled_d = float(parts[2])
                avg_speed = float(parts[3])
            except ValueError:
                continue

            i = 4

            while i + 5 < len(parts):

                try:

                    lat = float(parts[i])
                    lon = float(parts[i+1])
                    speed = float(parts[i+2])
                    lon_acc = float(parts[i+3])
                    lat_acc = float(parts[i+4])
                    time = float(parts[i+5])

                    records.append([
                        filepath.name,
                        track_id,
                        type,
                        traveled_d,
                        avg_speed,
                        lat,
                        lon,
                        speed,
                        lon_acc,
                        lat_acc,
                        time
                    ])

                except ValueError:
                    pass

                i += 6

    return pd.DataFrame(
        records,
        columns=[
            "source_file",
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
        ]
    )
    
def load_all_pneuma(folder_path):

    folder = Path(folder_path)

    all_dfs = []

    csv_files = list(folder.glob("*.csv"))

    print(f"\nFound {len(csv_files)} CSV files\n")

    for file in csv_files:

        print(f"Loading: {file.name}")

        df = load_pneuma(file)

        print(f"Records: {len(df):,}")

        all_dfs.append(df)

    final_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    return final_df

if __name__ == "__main__":

    DATASET_FOLDER = "pNEUMA_dataset"

    folder = Path(DATASET_FOLDER)

    csv_files = list(folder.glob("*.csv"))

    total_rows = 0
    max_columns = 0
    total_trajectory_points = 0
    
    unique_track_ids = set()

    print(f"Files found: {len(csv_files)}")

    for file in csv_files:

        with open(file, "r", encoding="utf8") as f:

            first_line = f.readline()

            cols = len(first_line.split(";"))

            rows = 0         
            for line in f:
                rows += 1
                parts = [x.strip() for x in line.strip().split(";")]
                if len(parts) > 0:
                    unique_track_ids.add(f"{file.name}_{parts[0]}")
                    
                if (len(parts) > 4):
                    trajectory_points = (len(parts) - 4) // 6
                    total_trajectory_points += trajectory_points    

        total_rows += rows
        max_columns = max(max_columns, cols)

        print(f"{file.name}: {rows:,} rows")

    print("\n========================")
    print("DATASET SUMMARY")
    print("========================")

    print(f"Rows    : {total_rows:,}")
    print(f"Columns : {max_columns}")
    print(f"Vehicle Trajectories: {len(unique_track_ids):,}")
    print(f"Trajectory Points  : {total_trajectory_points:,}")