import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Chốt chết đường dẫn vào kho ông muốn dùng
INPUT_DIR = "./dataset_engineered"

# Bảng từ điển chuẩn (Đã loại người đi bộ, ép chuẩn chữ thường)
PCE_DICT = {
    "motorcycle": 0.5,
    "bicycle": 0.5,
    "car": 1.0,
    "taxi": 1.0,
    "medium vehicle": 1.5,
    "heavy vehicle": 2.5,
    "bus": 2.5,
}

def run_advanced_kmeans():
    print(f"1. ĐANG NẠP VÀ TIỆT TRÙNG DỮ LIỆU TỪ: {INPUT_DIR} ...")
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    
    grid_stats_list = []
    
    for filename in csv_files:
        file_path = os.path.join(INPUT_DIR, filename)
        
        # Chỉ đọc đúng 6 cột gốc cần thiết để siêu tiết kiệm RAM
        df = pd.read_csv(file_path, usecols=[
            'grid_id', 'lat', 'lon', 'unique_track_id', 'speed', 'type'
        ])
        
        # =========================================================
        # MÀNG LỌC AN TOÀN NGAY TRONG RAM (Chấp mọi lỗi của Step 1 cũ)
        # =========================================================
        # Chuẩn hóa tên xe: gọt khoảng trắng thừa + hạ xuống chữ thường
        norm_type = df['type'].astype(str).str.strip().str.lower()
        
        # Tiêu diệt Pedestrian (Người đi bộ)
        valid_mask = norm_type != 'pedestrian'
        df = df[valid_mask].copy()
        norm_type = norm_type[valid_mask]
        
        # Tự động map lại hệ số PCE chuẩn xác
        df['pce_factor'] = norm_type.map(PCE_DICT).fillna(1.0)
        df['is_crawling'] = (df['speed'] < 5.0).astype(int)

        # --- [GOM NHÓM TẦNG 1: TRONG NỘI BỘ FILE] ---
        agg_df = df.groupby('grid_id').agg(
            lat=('lat', 'mean'),
            lon=('lon', 'mean'),
            vehicle_count=('unique_track_id', 'nunique'), # Mật độ xe duy nhất
            grid_avg_speed=('speed', 'mean'),             # Tốc độ trung bình
            pce_volume=('pce_factor', 'sum'),             # Tải trọng mặt đường
            crawling_count=('is_crawling', 'sum')         # Số lượt bò lết
        ).reset_index()
        
        grid_stats_list.append(agg_df)

    # --- [GOM NHÓM TẦNG 2: GỘP 20 FILE THÀNH BẢN ĐỒ TỔNG] ---
    print("-> Đang hợp nhất 20 phân vùng thành bản đồ Athens...")
    full_grid_df = pd.concat(grid_stats_list)
    
    df_hotspot = full_grid_df.groupby('grid_id').agg(
        lat=('lat', 'mean'),
        lon=('lon', 'mean'),
        total_vehicles=('vehicle_count', 'sum'),      # Tổng lưu lượng xe (Traffic Density)
        avg_speed=('grid_avg_speed', 'mean'),         # Vận tốc trung bình thực tế
        total_pce_volume=('pce_volume', 'sum'),       # Tổng tải trọng quy đổi
        total_crawling=('crawling_count', 'sum')
    ).reset_index()

    print(f"=> Xong! Bản đồ thu được: {len(df_hotspot)} ô lưới mặt đường (Grids).\n")

    # ==========================================
    # 2. CHUẨN BỊ 5 FEATURES & CHUẨN HÓA (SCALING)
    # ==========================================
    print("2. ĐANG CHUẨN HÓA Z-SCORE (STANDARD SCALER)...")
    # Đưa đúng 5 đặc trưng ông đã chốt trong Report 4 vào
    features = ['lat', 'lon', 'total_vehicles', 'avg_speed', 'total_pce_volume']
    X = df_hotspot[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================
    # 3. CHẠY K-MEANS (K=3)
    # ==========================================
    print("3. ĐANG FIT MÔ HÌNH K-MEANS...")
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_hotspot['cluster'] = kmeans.fit_predict(X_scaled)

    # ==========================================
    # 4. CHẤM ĐIỂM METRICS
    # ==========================================
    sil_score = silhouette_score(X_scaled, df_hotspot['cluster'])
    db_score = davies_bouldin_score(X_scaled, df_hotspot['cluster'])

    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ K-MEANS (EVALUATION)")
    print("="*50)
    print(f" - Số lượng cụm (K)    : {k}")
    print(f" - Silhouette Score    : {sil_score:.4f} (Gần 1 -> Rất tốt)")
    print(f" - Davies-Bouldin Index: {db_score:.4f} (Gần 0 -> Chặt chẽ)")
    
    # ==========================================
    # 5. GIẢI MÃ CỤM NÀO LÀ "HOTSPOT"
    # ==========================================
    print("\nĐẶC ĐIỂM CÁC CỤM (CLUSTER PROFILING):")
    cluster_summary = df_hotspot.groupby('cluster').agg(
        Số_ô_lưới=('grid_id', 'count'),
        Lưu_lượng_xe_TB=('total_vehicles', 'mean'),
        Tải_trọng_PCE_TB=('total_pce_volume', 'mean'),
        Tốc_độ_TB=('avg_speed', 'mean')
    ).round(2).sort_values(by='Tải_trọng_PCE_TB', ascending=False)
    
    print(cluster_summary)

    # ==========================================
    # 6. XUẤT BẢN ĐỒ KẸT XE
    # ==========================================
    print("\n6. ĐANG VẼ BẢN ĐỒ...")
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=df_hotspot, 
        x='lon', y='lat', 
        hue='cluster', 
        size='total_pce_volume', sizes=(15, 700), 
        palette='Set1', alpha=0.75, edgecolor='white', linewidth=0.3
    )
    
    plt.title('K-Means Clustering: Athens Congestion Hotspots', fontsize=15, fontweight='bold')
    plt.xlabel('Longitude (Kinh độ)', fontsize=12)
    plt.ylabel('Latitude (Vĩ độ)', fontsize=12)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Phân cụm")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('kmeans_hotspot_map.png', dpi=300)
    df_hotspot.to_csv('hotspots_result.csv', index=False)
    
    print("->  Đã lưu bản đồ kẹt xe: 'kmeans_hotspot_map.png'")
    print("->  Đã xuất danh sách tọa độ kẹt xe ra: 'hotspots_result.csv'")
    plt.show()

if __name__ == "__main__":
    run_advanced_kmeans()