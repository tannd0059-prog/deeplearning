import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Sinh chuỗi dữ liệu tổng hợp (Sóng Sine + Nhiễu Gaussian)
np.random.seed(42)
t = np.linspace(0, 50, 500)
raw_data = np.sin(t) + np.random.normal(0, 0.1, 500)

# 2. Xây dựng hàm tạo Cửa sổ trượt (Sliding Window)
def create_sliding_windows(data, window_size=15):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

WINDOW_SIZE = 15
X, y = create_sliding_windows(raw_data, window_size=WINDOW_SIZE)
X = X.unsqueeze(-1) # Chuyển dạng tensor sang [Batch, Seq_Len, Feature_Dim=1]

# 3. Mạng GRU cho bài toán Dự báo Chuỗi thời gian (Regression)
class TimeSeriesGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super(TimeSeriesGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]) # Lấy trạng thái ở time-step cuối cùng

# 4. Huấn luyện mô hình
model = TimeSeriesGRU(input_dim=1, hidden_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_history = []
EPOCHS = 40

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    predictions = model(X).squeeze(-1)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

# 5. Vẽ biểu đồ Loss và kết quả dự đoán
plt.figure(figsize=(12, 4))

# Đồ thị 1: Đường cong Loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='red', label='MSE Training Loss')
plt.title('Đường cong Loss qua các Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Đồ thị 2: Dự đoán vs Thực tế
plt.subplot(1, 2, 2)
with torch.no_grad():
    predicted_vals = model(X).squeeze(-1).numpy()
plt.plot(y.numpy()[:100], label='Thực tế (Ground Truth)', color='black', alpha=0.7)
plt.plot(predicted_vals[:100], label='Mô hình GRU Dự đoán', color='blue', linestyle='--')
plt.title('So sánh Kết quả Dự đoán (100 điểm đầu tiên)')
plt.xlabel('Thời gian')
plt.ylabel('Giá trị')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()