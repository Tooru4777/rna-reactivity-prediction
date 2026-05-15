import torch
import torch.nn as nn

# =====================================================================
# 教學筆記 (架構設計與生物學意義):
# 1. Convolutional Neural Network (CNN):
#    在 Andrew Ng 課程中，CNN 常用於影像辨識 (尋找邊緣、形狀)。
#    在生物資訊學中，1D CNN 就像是一個滑動視窗 (Sliding Window)，
#    沿著 RNA 序列滑動，目的是捕捉局部的 Motif (例如 hairpins, bulges 等短序列特徵)。
#
# 2. Long Short-Term Memory (LSTM):
#    RNN 的進階版。
#    LSTM 的「記憶機制」能讓網路記住前面讀過的序列特徵。
#    我們使用雙向 (Bi-directional) LSTM，代表網路會從左到右讀一次，再從右到左讀一次，
#    充分理解整條 RNA 的上下文結構。
# =====================================================================

class RNAReactivityPredictor(nn.Module):
    def __init__(self, input_dim=4, cnn_out_dim=64, lstm_hidden_dim=128, output_dim=1, dropout=0.3):
        """
        參數說明:
        - input_dim: 4 (代表 A, C, G, U 的 One-Hot Encoding)
        - cnn_out_dim: CNN 擷取出的特徵數量
        - lstm_hidden_dim: LSTM 的隱藏層大小 (記憶容量)
        - output_dim: 1 (預測 Reactivity 反應活性，1維度連續數值)
        - dropout: Dropout 比率，用來防止 Overfitting
        """
        super(RNAReactivityPredictor, self).__init__()
        
        # ---------------------------------------------------------
        # 第一層: 1D Convolutional Layer (局部特徵萃取)
        # ---------------------------------------------------------
        self.cnn = nn.Conv1d(in_channels=input_dim, 
                             out_channels=cnn_out_dim, 
                             kernel_size=5, 
                             padding=2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # ---------------------------------------------------------
        # 第二層: Bi-directional LSTM (長距離上下文關聯)
        # ---------------------------------------------------------
        self.lstm = nn.LSTM(input_size=cnn_out_dim, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout)
        
        # ---------------------------------------------------------
        # 第三層: Fully Connected Layer (活性預測映射)
        # ---------------------------------------------------------
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        輸入 x 的形狀: (Batch_Size, Max_Length, 4)
        """
        x = x.transpose(1, 2)
        x = self.relu(self.cnn(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

if __name__ == "__main__":
    print("初始化 RNAReactivityPredictor 神經網路...")
    model = RNAReactivityPredictor()
    dummy_input = torch.randn(1, 200, 4)
    print(f"輸入的 RNA Tensor 形狀: {dummy_input.shape}")
    predictions = model(dummy_input)
    print(f"神經網路預測輸出的 Reactivity 形狀: {predictions.shape}")
    print("架構測試成功！準備進入 Training Pipeline。")
