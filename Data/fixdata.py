import pandas as pd

def encode_all_categorical_columns(input_file, output_file):
    print(f"Đang đọc dữ liệu từ: {input_file}")
    df = pd.read_csv(input_file)
    
    # 1. Xử lý các biến Nhị phân (Binary) -> 1 và 0
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
        print("- Đã số hóa cột Sex")

    if 'ExerciseAngina' in df.columns:
        df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
        print("- Đã số hóa cột ExerciseAngina")

    # 2. Xử lý các biến đa danh mục (One-Hot Encoding)
    # Đã thêm 'ChestPainType' vào danh sách này
    cols_to_dummy = []
    if 'RestingECG' in df.columns: cols_to_dummy.append('RestingECG')
    if 'ST_Slope' in df.columns: cols_to_dummy.append('ST_Slope')
    if 'ChestPainType' in df.columns: cols_to_dummy.append('ChestPainType')
        
    if cols_to_dummy:
        # Áp dụng get_dummies với drop_first=True
        df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)
        
        # Chuyển các cột True/False sinh ra về dạng số nguyên (1/0)
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
        print(f"- Đã áp dụng One-Hot Encoding cho các cột: {cols_to_dummy}")

    # 3. Lưu kết quả ra file mới
    df.to_csv(output_file, index=False)
    print(f"\n[Thành công] Đã lưu dữ liệu sau khi số hóa tại: {output_file}")
    
    return df

# ==========================================
# CÁCH CHẠY THỬ
# ==========================================
if __name__ == "__main__":
    INPUT = 'heart.csv'
    OUTPUT = 'heart_encoded_final.csv'
    
    df_result = encode_all_categorical_columns(INPUT, OUTPUT)
    
    # In ra danh sách tất cả các cột hiện tại để bạn kiểm tra
    print("\n--- DANH SÁCH CÁC CỘT SAU KHI SỐ HÓA ---")
    print(df_result.columns.tolist())
    
    # Xem thử 5 dòng đầu của các cột vừa được One-Hot
    print("\n--- XEM THỬ KẾT QUẢ CỘT CHEST_PAIN ---")
    chest_pain_cols = [c for c in df_result.columns if 'ChestPainType' in c]
    print(df_result[chest_pain_cols].head())