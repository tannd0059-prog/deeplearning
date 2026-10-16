import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. THIẾT LẬP CẤU HÌNH & THƯ MỤC KẾT QUẢ
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# Tự động tạo thư mục 'results' nếu chưa tồn tại
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Cố định random seed để tái lập kết quả
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 2. TẠO DỮ LIỆU SÓNG SINE & SLIDING WINDOW
# ==========================================
# Sinh 600 điểm dữ liệu chuỗi thời gian (Sóng Sine + Nhiễu Gaussian)
t = np.linspace(0, 50, 600)
raw_data = np.sin(t) + np.random.normal(0, 0.08, 600)

# Hàm tạo cửa sổ trượt (Sliding Window)
def create_sliding_windows(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

WINDOW_SIZE = 20
X, y = create_sliding_windows(raw_data, window_size=WINDOW_SIZE)
X = X.unsqueeze(-1)  # Kích thước Tensor: [Batch_Size, Seq_Len=20, Feature_Dim=1]

# Chia dữ liệu: 80% Train, 20% Test
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size].to(device), y[:train_size].to(device)
X_test, y_test = X[train_size:].to(device), y[train_size:].to(device)

print(f"Tổng số mẫu: {len(X)} | Tập Train: {len(X_train)} | Tập Test: {len(X_test)}")

# ==========================================
# 3. ĐỊNH NGHĨA MÔ HÌNH GRU (REGRESSION)
# ==========================================
class TimeSeriesGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, dropout=0.2):
        super(TimeSeriesGRU, self).__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        out, _ = self.gru(x)
        # Lấy trạng thái ẩn ở time-step cuối cùng: out[:, -1, :]
        out = self.fc(out[:, -1, :])
        return out

model = TimeSeriesGRU(input_dim=1, hidden_dim=32, num_layers=2, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
# ==========================================
EPOCHS = 60
train_losses = []
test_losses = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    # Phase Train
    model.train()
    optimizer.zero_grad()
    preds_train = model(X_train).squeeze(-1)
    loss_train = criterion(preds_train, y_train)
    loss_train.backward()
    optimizer.step()
    
    # Phase Test (Evaluation)
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test).squeeze(-1)
        loss_test = criterion(preds_test, y_test)
        
    train_losses.append(loss_train.item())
    test_losses.append(loss_test.item())
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{EPOCHS}] - Train Loss: {loss_train.item():.5f} | Test Loss: {loss_test.item():.5f}")

training_time = time.time() - start_time
print(f"\n---> Hoàn thành huấn luyện trong: {training_time:.2f} giây")

# ==========================================
# 5. TRỰC QUAN HÓA & LƯU VÀO THƯ MỤC RESULTS/
# ==========================================
plt.figure(figsize=(14, 5))

# Đồ thị 1: Đường cong Loss (Train vs Test Loss)
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train MSE Loss', color='blue', linewidth=2)
plt.plot(test_losses, label='Test MSE Loss', color='orange', linestyle='--', linewidth=2)
plt.title('Đường cong Loss qua các Epochs (GRU)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('MSE Loss', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Đồ thị 2: Dự đoán vs Thực tế trên Tập Test
plt.subplot(1, 2, 2)
model.eval()
with torch.no_grad():
    y_test_cpu = y_test.cpu().numpy()
    preds_test_cpu = model(X_test).squeeze(-1).cpu().numpy()

plt.plot(y_test_cpu, label='Thực tế (Ground Truth)', color='black', alpha=0.8, linewidth=1.5)
plt.plot(preds_test_cpu, label='GRU Dự đoán (Predicted)', color='red', linestyle='--', linewidth=2)
plt.title('Dự báo Chuỗi Thời gian trên Tập Test', fontsize=12, fontweight='bold')
plt.xlabel('Time Step (Tập Test)', fontsize=10)
plt.ylabel('Giá trị', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()

#  Lưu biểu đồ vào thư mục results/
plot_path = os.path.join(results_dir, 'practice3_gru_time_series.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[OK] Đã lưu đồ thị vào: {plot_path}")
plt.show()

# ==========================================
# 6. LƯU BÁO CÁO CÁC CHỈ SỐ RA FILE TEXT
# ==========================================
metrics_path = os.path.join(results_dir, 'practice3_metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("=== KẾT QUẢ THỰC NGHIỆM PRACTICE 3 (GRU TIME SERIES) ===\n")
    f.write(f"Số lượng Epochs             : {EPOCHS}\n")
    f.write(f"Cửa sổ trượt (Window Size)  : {WINDOW_SIZE}\n")
    f.write(f"Thời gian Huấn luyện        : {training_time:.2f}s\n")
    f.write(f"Train MSE Loss cuối cùng    : {train_losses[-1]:.6f}\n")
    f.write(f"Test MSE Loss cuối cùng     : {test_losses[-1]:.6f}\n")

print(f"[OK] Đã lưu báo cáo chỉ số vào: {metrics_path}")