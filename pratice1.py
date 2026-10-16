import torch
import torch.nn as nn
import torch.optim as optim
import time

# 1. Định nghĩa cấu trúc mạng linh hoạt
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, model_type='LSTM', bidirectional=False):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.model_type = model_type
        self.bidirectional = bidirectional
        
        # Lựa chọn kiến trúc theo tham số
        if model_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
            
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        out = self.embedding(x) # [batch_size, seq_len, embed_dim]
        
        if self.model_type == 'LSTM':
            _, (h_n, _) = self.rnn(out)
        else:
            _, h_n = self.rnn(out)
            
        if self.bidirectional:
            # Gộp trạng thái ẩn cuối cùng của chiều tiến và chiều lùi
            h_last = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_last = h_n[-1, :, :]
            
        return self.fc(h_last)

# 2. Tạo dữ liệu giả lập để huấn luyện
VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 2
BATCH_SIZE = 32
SEQ_LEN = 10

X_train = torch.randint(0, VOCAB_SIZE, (128, SEQ_LEN))
y_train = torch.randint(0, NUM_CLASSES, (128,))

# 3. Hàm huấn luyện và đánh giá mô hình
def train_and_eval(model_type, bidirectional=False):
    model = TextClassificationModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, model_type, bidirectional)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    print(f"Mô hình: {model_type}{' (Bi)' if bidirectional else ''} | Loss cuối: {loss.item():.4f} | Thời gian: {elapsed:.4f}s")

# Chạy thử nghiệm so sánh 3 kiến trúc
train_and_eval('RNN', bidirectional=False)
train_and_eval('GRU', bidirectional=False)
train_and_eval('LSTM', bidirectional=True)