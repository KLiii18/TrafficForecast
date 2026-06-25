import os
import time
import numpy as np
import pandas as pd

INPUT_DIR = "./dataset"
OUTPUT_DIR = "./dataset_engineered"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PCE_DICT = {
    "Motorcycle": 0.5,
    "Car": 1.0,
    "Taxi": 1.0,
    "Medium Vehicle": 1.5,
    "Heavy Vehicle": 2.5,
    "Bus": 2.5,
}


def batch_process():
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    total_files = len(csv_files)

    print(f"=== PHÁT LỆNH TỔNG TẤN CÔNG: 'ĐỘ' {total_files} FILE CSV ===")
    start_time = time.time()

    success_count = 0
    failed_files = []

    for idx, filename in enumerate(csv_files, 1):
        file_path = os.path.join(INPUT_DIR, filename)

        # Bóc tách mốc thời gian từ tên file
        parts = filename.replace(".csv", "").split("_")
        try:
            date_str = parts[1]  # '20181024'
            start_time_str = parts[3]  # '0830'
            base_ts = pd.to_datetime(
                f"{date_str} {start_time_str}", format="%Y%m%d %H%M"
            )
        except Exception as e:
            print(f"  [!] Bỏ qua {filename}: Lỗi format tên file ({e})")
            failed_files.append(filename)
            continue

        print(f"[{idx}/{total_files}] Đang cày: {filename}...")

        try:
            df = pd.read_csv(file_path)

            # --- NẠP 8 VŨ KHÍ ---
            df["timestamp_real"] = base_ts + pd.to_timedelta(
                df["time"], unit="s"
            )
            df["time_str"] = df["timestamp_real"].dt.strftime("%H:%M:%S")
            df["time_bin_5m"] = df["timestamp_real"].dt.floor("5min")

            df["unique_track_id"] = (
                f"{date_str}_{start_time_str}_" + df["track_id"].astype(str)
            )
            df["grid_id"] = (
                df["lat"].round(4).astype(str)
                + "_"
                + df["lon"].round(4).astype(str)
            )

            df["is_crawling"] = np.where(df["speed"] < 5.0, 1, 0)
            df["is_hard_braking"] = np.where(df["lon_acc"] < -3.0, 1, 0)
            df["pce_factor"] = df["type"].str.strip().map(PCE_DICT).fillna(1.0)

            # Xuất xưởng
            out_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(out_path, index=False)

            success_count += 1

        except Exception as e:
            print(f"  [X] Gãy gánh tại file {filename}: {e}")
            failed_files.append(filename)

    # --- BÁO CÁO KẾT QUẢ CHIẾN DỊCH ---
    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 50)
    print(f"🎉 CHIẾN DỊCH HOÀN TẤT TRONG {elapsed} GIÂY!")
    print(f"  - Thành công: {success_count}/{total_files} file")
    if failed_files:
        print(f"  - Thất bại ({len(failed_files)} file): {failed_files}")
    print("=" * 50)


if __name__ == "__main__":
    batch_process()