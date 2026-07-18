import pandas as pd
import numpy as np

# Đọc dữ liệu từ file
df = pd.read_csv("heart2.csv")

# Lọc các giá trị Cholesterol hợp lệ (lớn hơn 0)
non_zero_chol = df[df['Cholesterol'] > 0]['Cholesterol']

# Lấy index của các dòng có Cholesterol bằng 0
zero_indices = df[df['Cholesterol'] == 0].index

# Lấy mẫu ngẫu nhiên từ tập hợp lệ để thay thế các số 0
np.random.seed(42) # (Tùy chọn) Cố định seed để kết quả đồng nhất ở các lần chạy
random_samples = np.random.choice(non_zero_chol, size=len(zero_indices), replace=True)
df.loc[zero_indices, 'Cholesterol'] = random_samples

# Lưu ra file mới sau khi đã thay thế xong
df.to_csv("heart2_modified.csv", index=False)