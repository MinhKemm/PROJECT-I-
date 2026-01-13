import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# IO
def load_data():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # nếu file trong src/
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    return X, y

def save_model(model, name):
    joblib.dump(model, f'models/{name}.pkl')

def save_metrics(name, metrics_dict):
    formatted_metrics = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                         for k, v in metrics_dict.items()}
    
    try:
        if os.path.exists('results/metrics.json'):
            with open('results/metrics.json', 'r') as f:
                data = json.load(f)
        else:
            data = {}
    except (json.JSONDecodeError):
        data = {}
        
    data[name] = formatted_metrics
    
    os.makedirs('results', exist_ok=True)
    
    with open('results/metrics.json', 'w') as f:
        json.dump(data, f, indent=4)

# Metric
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# OOF (Out-of-Fold) cho Stacking
def get_oof_preds(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
        
    return oof_preds


def save_figure(fig, filename):
    """
    Lưu biểu đồ (matplotlib figure) vào thư mục results/figures/.
    Tạo thư mục nếu nó chưa tồn tại.
    """
    # Đường dẫn tuyệt đối đến thư mục figures
    FIGURES_PATH = Path("results/figures")
    
    # Đảm bảo thư mục tồn tại (tạo nếu chưa có)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    # Lưu hình ảnh
    fig.savefig(FIGURES_PATH / filename, bbox_inches='tight')
    print(f"Hình ảnh đã lưu thành công tại: {FIGURES_PATH / filename}")
