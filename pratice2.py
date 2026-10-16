import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gộp tất cả trọng số của 4 cổng thành 1 ma trận duy nhất để tối ưu tốc độ nhân ma trận
        # Kích thước ma trận trọng số: [4 * hidden_dim, input_dim] và [4 * hidden_dim, hidden_dim]
        self.W = nn.Parameter(torch.randn(4 * hidden_dim, input_dim) * 0.01)
        self.U = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_dim))

    def forward(self, x_t, h_prev, c_prev):
        # x_t shape: [batch_size, input_dim]
        # h_prev, c_prev shape: [batch_size, hidden_dim]
        
        # Tính tổng tuyến tính cho 4 cổng trong 1 phép nhân ma trận
        gates = torch.matmul(x_t, self.W.t()) + torch.matmul(h_prev, self.U.t()) + self.bias
        
        # Tách kết quả thành 4 phần bằng nhau tương ứng với 4 cổng
        f_gate, i_gate, c_cand, o_gate = torch.chunk(gates, 4, dim=1)
        
        # Áp dụng các hàm kích hoạt (Activation functions)
        f_t = torch.sigmoid(f_gate)       # Cổng quên (Forget gate)
        i_t = torch.sigmoid(i_gate)       # Cổng vào (Input gate)
        c_tilde = torch.tanh(c_cand)      # Cập nhật ứng viên
        o_t = torch.sigmoid(o_gate)       # Cổng ra (Output gate)
        
        # Tính toán trạng thái ô (Cell State) và trạng thái ẩn (Hidden State) mới
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

# Kiểm tra tính đúng đắn về kích thước dữ liệu (Shape Verification)
batch_size, input_dim, hidden_dim = 4, 10, 20
custom_cell = CustomLSTMCell(input_dim, hidden_dim)

x_dummy = torch.randn(batch_size, input_dim)
h_dummy = torch.zeros(batch_size, hidden_dim)
c_dummy = torch.zeros(batch_size, hidden_dim)

h_next, c_next = custom_cell(x_dummy, h_dummy, c_dummy)
print("Shape của Hidden State mới:", h_next.shape) # Kế thừa: [4, 20]
print("Shape của Cell State mới:", c_next.shape)   # Kế thừa: [4, 20]