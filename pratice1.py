import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. THIẾT LẬP CẤU HÌNH & THƯ MỤC KẾT QUẢ
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tự động tạo thư mục 'results'
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

torch.manual_seed(42)

# ==========================================
# 2. ĐỊNH NGHĨA MÔ HÌNH THẮNG CẢ 3 KIẾN TRÚC
# ==========================================
class TextClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        model_type="LSTM",
        bidirectional=False,
    ):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.model_type = model_type
        self.bidirectional = bidirectional

        if model_type == "RNN":
            self.rnn = nn.RNN(
                embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
            )
        elif model_type == "GRU":
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
            )
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
            )

        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        if self.model_type == "LSTM":
            _, (h_n, _) = self.rnn(out)
        else:
            _, h_n = self.rnn(out)

        if self.bidirectional:
            h_last = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_last = h_n[-1, :, :]

        return self.fc(h_last)


# ==========================================
# 3. TẠO DỮ LIỆU GIẢ LẬP (TRAIN & TEST)
# ==========================================
VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 2
SEQ_LEN = 12

X_train = torch.randint(0, VOCAB_SIZE, (256, SEQ_LEN)).to(device)
y_train = torch.randint(0, NUM_CLASSES, (256,)).to(device)
X_test = torch.randint(0, VOCAB_SIZE, (64, SEQ_LEN)).to(device)
y_test = torch.randint(0, NUM_CLASSES, (64,)).to(device)

# ==========================================
# 4. HUẤN LUYỆN & SO SÁNH 3 MÔ HÌNH
# ==========================================
EPOCHS = 20
models_to_test = [
    ("Vanilla RNN", "RNN", False),
    ("GRU", "GRU", False),
    ("BiLSTM", "LSTM", True),
]

history = {}

for name, model_type, bi in models_to_test:
    print(f"\n---> Đang huấn luyện: {name}...")
    model = TextClassificationModel(
        VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, model_type, bi
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    loss_epoch = []
    acc_epoch = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Đánh giá Accuracy trên tập Test
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            acc = (
                (test_preds.argmax(dim=1) == y_test).float().mean().item() * 100
            )

        loss_epoch.append(loss.item())
        acc_epoch.append(acc)

    elapsed_time = time.time() - start_time
    history[name] = {
        "loss": loss_epoch,
        "acc": acc_epoch,
        "time": elapsed_time,
        "final_loss": loss_epoch[-1],
        "final_acc": acc_epoch[-1],
    }
    print(
        f"Hoàn thành {name} | Final Loss: {loss_epoch[-1]:.4f} | Test Acc: {acc_epoch[-1]:.2f}% | Time: {elapsed_time:.2f}s"
    )

# ==========================================
# 5. TRỰC QUAN HÓA & LƯU HÌNH VÀO RESULTS/
# ==========================================
plt.figure(figsize=(13, 5))

# Đồ thị 1: So sánh Đường cong Loss
plt.subplot(1, 2, 1)
colors = {"Vanilla RNN": "red", "GRU": "green", "BiLSTM": "blue"}
for name in history:
    plt.plot(
        history[name]["loss"],
        label=name,
        color=colors[name],
        linewidth=2,
    )
plt.title(
    "So sánh Đường cong Training Loss giữa các kiến trúc",
    fontsize=11,
    fontweight="bold",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# Đồ thị 2: So sánh Độ chính xác Accuracy (%)
plt.subplot(1, 2, 2)
for name in history:
    plt.plot(
        history[name]["acc"],
        label=name,
        color=colors[name],
        linestyle="--",
        linewidth=2,
    )
plt.title(
    "So sánh Độ chính xác Test Accuracy (%) qua các Epochs",
    fontsize=11,
    fontweight="bold",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plot_path = os.path.join(results_dir, "practice1_sentiment_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\n[OK] Đã lưu đồ thị vào: {plot_path}")
plt.show()

# ==========================================
# 6. LƯU BÁO CÁO CÁC CHỈ SỐ RA FILE TEXT
# ==========================================
metrics_path = os.path.join(results_dir, "practice1_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(
        "=== BÁO CÁO SO SÁNH HIỆU NĂNG CÁC KIẾN TRÚC RNN (PRACTICE 1) ===\n\n"
    )
    f.write(
        f"{'Kiến trúc':<15} | {'Final Loss':<12} | {'Test Acc (%)':<14} | {'Thời gian (s)':<12}\n"
    )
    f.write("-" * 60 + "\n")
    for name, data in history.items():
        f.write(
            f"{name:<15} | {data['final_loss']:<12.4f} | {data['final_acc']:<14.2f} | {data['time']:<12.2f}\n"
        )

print(f"[OK] Đã lưu báo cáo chỉ số vào: {metrics_path}")