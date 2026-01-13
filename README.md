## 🏡 Dự báo Giá Nhà bằng Machine Learning

Dự án này nhằm xây dựng một hệ thống học máy dự báo giá nhà dựa trên tập dữ liệu bất động sản. Mục tiêu là so sánh hiệu năng của nhiều mô hình, từ các mô hình hồi quy cơ bản đến các phương pháp *Ensemble Learning*, và xác định mô hình cho hiệu suất dự báo tốt nhất.

---

## 📂 Cấu trúc thư mục 
Dưới đây là sơ đồ tổ chức của dự án:

```plaintext
PROJECT-I/
├── data/
│   ├── raw/                # Dữ liệu gốc từ Kaggle (train.csv, test.csv,...)
│   └── processed/          # Dữ liệu sau khi xử lý (X_train.csv, y_train.csv,...)
├── demo/
│   ├── results/            # Lưu trữ kết quả chạy demo
│   └── Demo.ipynb          
├── models/                 # Chứa các mô hình đã huấn luyện (.pkl)
├── notebooks/              # Các bước thực nghiệm chi tiết
│   ├── eda.ipynb           # Phân tích dữ liệu khám phá
│   ├── preprocessing.ipynb # Tiền xử lý & Feature Engineering
│   ├── linear.ipynb        # Huấn luyện chi tiết từng loại mô hình
│   ├── ....
│   └── stacking_model.ipynb 
├── results/
│   ├── figures/            # Biểu đồ phần dư, Biểu đồ Predicted vs Actual
│   └── metrics.json        # Tổng hợp các chỉ số RMSE, MAE, R2 của các mô hình
├── src/                    # Mã nguồn module hóa
│   ├── models.py           # Định nghĩa cấu trúc các lớp mô hình
│   ├── tuning.py           # Scripts tối ưu hóa siêu tham số
│   └── utils.py            # Các hàm bổ trợ xử lý dữ liệu
├── .gitignore             
├── README.md               
└── requirements.txt        
```

---

## 🚀 Hướng dẫn cài đặt và Sử dụng

### 📥 1. Clone repository

```bash
git clone https://github.com/MinhKemm/PROJECT-I-.git
cd PROJECT-I-
```

---

### 📦 2. Thiết lập môi trường
Sử dụng venv

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate
# Kích hoạt môi trường (macOS/Linux)
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```
---

### 🚀 3. Cách sử dụng
Bước 1: Chạy `notebooks/eda.ipynb` để hiểu về dữ liệu.

Bước 2: Chạy `notebooks/preprocessing.ipynb` để tạo bộ dữ liệu sạch trong `data/processed`.

Bước 3: Chạy các file `notebook` trong `notebooks/` để huấn luyện mô hình.

Bước 4: Để xem kết quả dự báo nhanh, mở `demo/Demo.ipynb`.
