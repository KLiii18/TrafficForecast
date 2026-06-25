import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
import seaborn as sns

def verify_kmeans_decisions():
    print("1. ĐANG ĐỌC KẾT QUẢ K-MEANS TỪ FILE 'hotspots_result.csv'...")
    try:
        df = pd.read_csv('hotspots_result.csv')
    except FileNotFoundError:
        print("[!] Lỗi: Không tìm thấy file 'hotspots_result.csv'. Ông hãy chạy lại step2_kmeans_advanced.py trước nhé!")
        return

    features = ['total_vehicles', 'avg_speed', 'total_pce_volume', 'total_crawling']

    # =========================================================
    # CÁCH 1: SOI "ADN TÂM CỤM" (CLUSTER CENTROID PROFILING)
    # =========================================================
    print("\n" + "="*55)
    print("🔍 [CÁCH 1] BẢNG 'ADN' ĐẶC TRƯNG CỦA TỪNG CỤM")
    print("="*55)
    
    profile = df.groupby('cluster')[features].mean().round(2)
    print(profile)
    
    # =========================================================
    # CÁCH 2: "TRỒNG CÂY NGƯỢC" ĐỂ GIẢI MÃ LUẬT IF-ELSE
    # =========================================================
    print("\n" + "="*55)
    print("🌳 [CÁCH 2] GIẢI MÃ LUẬT K-MEANS BẰNG CÂY QUYẾT ĐỊNH")
    print("="*55)
    
    X = df[['total_pce_volume', 'avg_speed', 'total_vehicles']]
    y = df['cluster']
    
    # Huấn luyện cây quyết định độ sâu tối đa = 3 để con người dễ đọc
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    tree_rules = export_text(tree, feature_names=list(X.columns))
    print("Luật phân loại máy tính đã ngầm định sử dụng:")
    print(tree_rules)

    # =========================================================
    # CÁCH 3: KIỂM CHỨNG ĐỐI CHIẾU VỆ TINH (GROUND TRUTH)
    # =========================================================
    print("\n" + "="*55)
    print("🛰️ [CÁCH 3] LINK ĐỐI CHIẾU ẢNH VỆ TINH GOOGLE MAPS")
    print("="*55)
    
    # Lấy ra cụm có tải trọng lớn nhất (Đại lộ) và nhỏ nhất (Hẻm)
    sorted_clusters = profile.sort_values(by='total_pce_volume', ascending=False).index
    highway_cluster = sorted_clusters[0] # Cụm Đỏ
    alley_cluster = sorted_clusters[-1]  # Cụm Xanh
    
    print(f"\n--- Top 3 ô lưới tiêu biểu của CỤM {highway_cluster} (Dự đoán là ĐẠI LỘ) ---")
    top_highways = df[df['cluster'] == highway_cluster].sort_values(by='total_pce_volume', ascending=False).head(3)
    for idx, row in top_highways.iterrows():
        url = f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"
        print(f" * Grid [{row['grid_id']}] | Lưu lượng PCE: {row['total_pce_volume']:,.0f} | Link vệ tinh: {url}")

    print(f"\n--- Top 3 ô lưới tiêu biểu của CỤM {alley_cluster} (Dự đoán là ĐƯỜNG HẺM) ---")
    top_alleys = df[df['cluster'] == alley_cluster].sort_values(by='total_pce_volume', ascending=True).head(3)
    for idx, row in top_alleys.iterrows():
        url = f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"
        print(f" * Grid [{row['grid_id']}] | Lưu lượng PCE: {row['total_pce_volume']:,.0f} | Link vệ tinh: {url}")

    # =========================================================
    # CÁCH 4: VẼ BOXPLOT ĐỂ CHỨNG MINH RÕ RÀNG TRÊN TRỤC SỐ
    # =========================================================
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='cluster', y='total_pce_volume', palette='Set1', showfliers=False)
    plt.title('Sự phân hóa về Tải trọng xe (PCE Volume)', fontweight='bold')
    plt.ylabel('PCE Volume')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='cluster', y='avg_speed', palette='Set1', showfliers=False)
    plt.title('Sự phân hóa về Tốc độ trung bình (km/h)', fontweight='bold')
    plt.ylabel('Average Speed (km/h)')
    
    plt.tight_layout()
    plt.savefig('kmeans_proof_boxplots.png', dpi=300)
    print("\n[OK] Đã xuất biểu đồ chứng minh phân hóa ra file 'kmeans_proof_boxplots.png'")
    plt.show()

if __name__ == "__main__":
    verify_kmeans_decisions()