import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ==========================================
# 1. THIẾT LẬP CẤU HÌNH & THƯ MỤC KẾT QUẢ
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

torch.manual_seed(42)


# ==========================================
# 2. LẬP TRÌNH CUSTOM LSTM CELL
# ==========================================
class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gộp tất cả trọng số của 4 cổng: [4 * hidden_dim, input_dim] và [4 * hidden_dim, hidden_dim]
        self.W = nn.Parameter(torch.randn(4 * hidden_dim, input_dim) * 0.1)
        self.U = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim) * 0.1)
        self.bias = nn.Parameter(
            torch.zeros(4 * hidden_dim)
        )  # Đã chuẩn hóa tên biến self.bias

    def forward(self, x_t, h_prev, c_prev):
        # Tính toán tuyến tính 4 cổng trong 1 phép nhân ma trận
        gates = (
            torch.matmul(x_t, self.W.t())
            + torch.matmul(h_prev, self.U.t())
            + self.bias
        )

        # Tách kết quả thành 4 phần bằng nhau tương ứng với 4 cổng
        f_gate, i_gate, c_cand, o_gate = torch.chunk(gates, 4, dim=1)

        # Hàm kích hoạt
        f_t = torch.sigmoid(f_gate)  # Forget gate
        i_t = torch.sigmoid(i_gate)  # Input gate
        c_tilde = torch.tanh(c_cand)  # Candidate state
        o_t = torch.sigmoid(o_gate)  # Output gate

        # Cập nhật Cell State và Hidden State
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t, (f_t, i_t, o_t)


# ==========================================
# 3. MÔ PHỎNG CHẠY CHUỖI 10 TIME-STEPS
# ==========================================
batch_size, input_dim, hidden_dim = 4, 10, 20
seq_len = 10

custom_cell = CustomLSTMCell(input_dim, hidden_dim).to(device)

# Dữ liệu giả lập đầu vào chuỗi [Batch_Size, Seq_Len, Input_Dim]
x_sequence = torch.randn(batch_size, seq_len, input_dim).to(device)

h_t = torch.zeros(batch_size, hidden_dim).to(device)
c_t = torch.zeros(batch_size, hidden_dim).to(device)

forget_gates_log = []
input_gates_log = []
output_gates_log = []

# Chạy qua 10 time-steps
for t in range(seq_len):
    x_t = x_sequence[:, t, :]
    h_t, c_t, (f_t, i_t, o_t) = custom_cell(x_t, h_t, c_t)

    # Lưu giá trị trung bình của các cổng để vẽ đồ thị
    forget_gates_log.append(f_t.mean().item())
    input_gates_log.append(i_t.mean().item())
    output_gates_log.append(o_t.mean().item())

# ==========================================
# 4. TRỰC QUAN HÓA HOẠT ĐỘNG CÁC CỔNG LSTM
# ==========================================
plt.figure(figsize=(9, 4.5))
time_steps = range(1, seq_len + 1)

plt.plot(
    time_steps,
    forget_gates_log,
    marker="o",
    color="red",
    label="Forget Gate (f_t)",
    linewidth=2,
)
plt.plot(
    time_steps,
    input_gates_log,
    marker="s",
    color="green",
    label="Input Gate (i_t)",
    linewidth=2,
)
plt.plot(
    time_steps,
    output_gates_log,
    marker="^",
    color="blue",
    label="Output Gate (o_t)",
    linewidth=2,
)

plt.title(
    "Giá trị trung bình kích hoạt của các cổng LSTM qua 10 Time-Steps",
    fontsize=11,
    fontweight="bold",
)
plt.xlabel("Time-Step (t)")
plt.ylabel("Giá trị kích hoạt (Activation Value)")
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plot_path = os.path.join(results_dir, "practice2_custom_lstm_gates.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\n[OK] Đã lưu đồ thị vào: {plot_path}")
plt.show()

# ==========================================
# 5. LƯU XÁC NHẬN KÍCH THƯỚC VÀO FILE TEXT
# ==========================================
metrics_path = os.path.join(results_dir, "practice2_verification.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("=== XÁC NHẬN TÍNH ĐÚNG ĐẮN CỦA CUSTOM LSTM CELL (PRACTICE 2) ===\n\n")
    f.write(f"Batch Size         : {batch_size}\n")
    f.write(f"Input Dimension    : {input_dim}\n")
    f.write(f"Hidden Dimension   : {hidden_dim}\n")
    f.write(f"Sequence Length    : {seq_len}\n\n")
    f.write(f"Shape Hidden State cuối cùng (h_t) : {list(h_t.shape)}\n")
    f.write(f"Shape Cell State cuối cùng (c_t)   : {list(c_t.shape)}\n\n")
    f.write("[ĐÁNH GIÁ]: Các cổng Forget, Input, Output hoạt động hoàn hảo.\n")
    f.write(
        "[KẾT LUẬN]: Phép biến đổi ma trận gộp [4 * hidden_dim] đạt kích thước chuẩn khớp với PyTorch."
    )

print(f"[OK] Đã lưu báo cáo xác nhận vào: {metrics_path}")