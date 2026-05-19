import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os

try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    print("\n[嚴重警告] 找不到 ViennaRNA 套件！")
    print("請在 Kaggle Notebook 最上方的 Cell 執行以下指令進行安裝：")
    print("!pip install viennarna")
    print("目前將退回使用無結構的 Dummy 模式，這會導致模型學不到真實的 2D 結構。\n")
    HAS_VIENNA = False

# =====================================================================
# 模組 0: Model Architecture (將模型直接寫在此處，方便在 Kaggle 上執行)
# =====================================================================
class RNAReactivityPredictor(nn.Module):
    # 將預設的 input_dim 從 4 (ACGU) 改為 7 (ACGU + 括號左, 括號右, 小數點)
    def __init__(self, input_dim=7, cnn_out_dim=128, lstm_hidden_dim=128, 
                 transformer_nhead=8, transformer_layers=2, output_dim=2, dropout=0.3):
        super(RNAReactivityPredictor, self).__init__()
        
        # 1. CNN: 擷取局部特徵 (雙層 + BatchNorm)
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_out_dim // 2)
        self.cnn2 = nn.Conv1d(in_channels=cnn_out_dim // 2, out_channels=cnn_out_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_out_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 2. Bi-LSTM: 序列關聯與時序位置編碼
        self.lstm = nn.LSTM(input_size=cnn_out_dim, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout)
                            
        # 3. Transformer Encoder: 捕捉長距離交互作用 (Self-Attention)
        d_model = lstm_hidden_dim * 2 # Bi-LSTM 的輸出維度是 256
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=transformer_nhead, 
                                                   dim_feedforward=d_model*4, 
                                                   dropout=dropout, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 4. 輸出層
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # 1. CNN 處理
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        
        # 2. LSTM 處理 (兼作 Positional Encoding)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # 3. Transformer 處理 (加入 Residual Connection 穩定訓練)
        # transformer_out shape: (batch_size, seq_len, d_model)
        transformer_out = self.transformer(lstm_out)
        transformer_out = transformer_out + lstm_out
        
        # 4. 預測輸出
        out = self.fc(transformer_out)
        return out

# =====================================================================
# 模組 1: Data Pipeline (資料前處理)
# =====================================================================
class RNAReactivityDataset(Dataset):
    def __init__(self, sequences_csv, max_length=206):
        self.max_length = max_length
        # 維度 0~3 是序列，4~6 是結構
        self.char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '(': 4, ')': 5, '.': 6}
        
        if os.path.exists(sequences_csv):
            print(f"成功載入 Kaggle 真實資料: {sequences_csv}")
            self.seq_df = pd.read_csv(sequences_csv)
            
            # 過濾高品質資料 (Signal-to-Noise >= 1.0)
            if 'SN_filter' in self.seq_df.columns:
                initial_count = len(self.seq_df)
                self.seq_df = self.seq_df[self.seq_df['SN_filter'] == 1.0]
                print(f"過濾高品質資料 (SN_filter == 1.0): {initial_count} -> {len(self.seq_df)} 筆")
                
            # 將資料量提升到 50,000 筆，增加訓練樣本
            self.seq_df = self.seq_df.head(50000).reset_index(drop=True) 
            self.mock_data = False
            
            print("🚀 正在預先計算 RNA 2D 二級結構 (使用 ViennaRNA)... 這可能會需要一分鐘時間。")
            if HAS_VIENNA:
                # 計算結構並存入 DataFrame 中
                self.seq_df['structure'] = self.seq_df['sequence'].apply(lambda seq: RNA.fold(seq)[0])
            else:
                # 如果沒有套件，只能給一堆點當作 Dummy 結構
                self.seq_df['structure'] = self.seq_df['sequence'].apply(lambda seq: '.' * len(seq))
            print("✅ 結構計算完成！")
            self.num_samples = len(self.seq_df)
            
            # 找出所有的 reactivity 欄位 (排除 error 欄位)
            self.reactivity_cols = [c for c in self.seq_df.columns if c.startswith('reactivity_') and 'error' not in c]
            print(f"找到 {len(self.reactivity_cols)} 個 reactivity 欄位")
        else:
            print("[警告] 找不到 Kaggle CSV 檔案，使用模擬資料。")
            self.mock_data = True
            self.num_samples = 1000
            self.reactivity_cols = []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mock_data:
            length = np.random.randint(50, self.max_length)
            seq_str = ''.join(np.random.choice(['A', 'C', 'G', 'U'], size=length))
            # 產生模擬的 2D Reactivity 與 Mask (分別對應 2A3 與 DMS)
            reactivities = np.random.randn(length, 2).astype(np.float32)
            valid_mask = np.ones((length, 2), dtype=np.float32)
        else:
            seq_str = self.seq_df.iloc[idx].get('sequence', '')
            if pd.isna(seq_str) or len(seq_str) == 0:
                seq_str = 'A' * 50
            length = len(seq_str)
            
            # 動態提取真實的 Reactivity (處理 NaN)
            if len(self.reactivity_cols) > 0:
                row_reactivities = self.seq_df.iloc[idx][self.reactivity_cols].values
                actual_reactivity_len = min(length, len(row_reactivities))
                
                reactivities = np.zeros((length, 2), dtype=np.float32)
                valid_mask = np.zeros((length, 2), dtype=np.float32)
                
                # 根據實驗類型決定寫入哪個 Channel (0: 2A3, 1: DMS)
                experiment_type = self.seq_df.iloc[idx].get('experiment_type', '2A3_MaP')
                exp_idx = 0 if experiment_type == '2A3_MaP' else 1
                
                for i in range(actual_reactivity_len):
                    val = row_reactivities[i]
                    if not pd.isna(val): # 如果不是 NaN，代表是有效的測試值
                        reactivities[i, exp_idx] = float(val)
                        valid_mask[i, exp_idx] = 1.0 # 標記此核苷酸的數值有效，應被計算 Loss
            else:
                reactivities = np.zeros((length, 2), dtype=np.float32)
                valid_mask = np.zeros((length, 2), dtype=np.float32)

        seq_str = seq_str[:self.max_length]
        
        # 取得結構字串
        if self.mock_data:
            struct_str = '.' * len(seq_str)
        else:
            struct_str = self.seq_df.iloc[idx].get('structure', '.' * len(seq_str))[:self.max_length]
            
        reactivities = reactivities[:self.max_length]
        valid_mask = valid_mask[:self.max_length]
        actual_length = len(seq_str)

        # 變成 7 維度的 One-Hot (4維度給序列，3維度給結構)
        one_hot = np.zeros((self.max_length, 7), dtype=np.float32)
        for i in range(actual_length):
            char_seq = seq_str[i]
            char_struct = struct_str[i]
            if char_seq in self.char_map:
                one_hot[i, self.char_map[char_seq]] = 1.0
            if char_struct in self.char_map:
                one_hot[i, self.char_map[char_struct]] = 1.0

        padded_reactivities = np.zeros((self.max_length, 2), dtype=np.float32)
        padded_reactivities[:actual_length] = reactivities
        
        mask = np.zeros((self.max_length, 2), dtype=np.float32)
        mask[:actual_length] = valid_mask

        return torch.tensor(one_hot), torch.tensor(padded_reactivities), torch.tensor(mask)

# =====================================================================
# 模組 3: Training Loop (訓練流程)
# =====================================================================
def train_model():
    print("=== Kaggle Notebook 訓練啟動 (Ribonanza Reactivity 預測) ===")
    
    possible_paths = [
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/OLD/train_data.csv",
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/train_data.csv",
        "/kaggle/input/stanford-ribonanza-rna-folding/train_data.csv"
    ]
    
    KAGGLE_CSV_PATH = ""
    for path in possible_paths:
        if os.path.exists(path):
            KAGGLE_CSV_PATH = path
            break
            
    if not KAGGLE_CSV_PATH:
        # 當在本機找不到 Kaggle 巨大檔案時，改讀取這 1000 筆真實資料！
        KAGGLE_CSV_PATH = "train_data_1000.csv" 

    
    dataset = RNAReactivityDataset(sequences_csv=KAGGLE_CSV_PATH, max_length=206)
    
    test_size = int(0.15 * len(dataset))
    cv_size = int(0.15 * len(dataset))
    train_size = len(dataset) - cv_size - test_size
    train_dataset, cv_dataset, test_dataset = random_split(dataset, [train_size, cv_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"資料集總數: {len(dataset)} | 訓練集: {train_size} | CV集: {cv_size} | 測試集: {test_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練設備: {device}")
    
    # 初始化模型，並確認維度是 7
    print("模型架構: CNN + Bi-LSTM + Transformer (7 維輸入: ACGU + 2D 結構)")
    model = RNAReactivityPredictor(input_dim=7).to(device)
    # 改用 L1Loss (MAE)，這對 Kaggle 競賽中的 Outliers 比較強健
    criterion = nn.L1Loss(reduction='none') 
    # 加入 Weight Decay (L2 正則化) 來防止過度擬合
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # 加入 Learning Rate Scheduler，當 CV Loss 停滯時自動降低 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    num_epochs = 50 
    best_cv_loss = float('inf') 
    early_stop_patience = 7
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (sequences, reactivities, masks) in enumerate(train_loader):
            sequences, reactivities, masks = sequences.to(device), reactivities.to(device), masks.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences) 
            
            # 解決梯度消失 (Gradient Vanishing) 問題:
            # 訓練時不能把 predictions 也 clamp，否則超出 [0, 1] 範圍的預測值梯度會變成 0，模型無法學習將其拉回。
            # 正確作法：僅將目標值 (reactivities) clip 到 [0, 1]，然後計算與 predictions 的 L1 誤差，保留完整梯度。
            reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
            loss_matrix = criterion(predictions, reacts_clipped)
            masked_loss = loss_matrix * masks
            
            actual_nucleotides_count = masks.sum()
            if actual_nucleotides_count > 0:
                final_loss = masked_loss.sum() / actual_nucleotides_count
            else:
                final_loss = torch.tensor(0.0, requires_grad=True).to(device)
            
            final_loss.backward()
            optimizer.step()
            
            total_train_loss += final_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        
        model.eval()
        total_cv_loss = 0.0
        with torch.no_grad():
            for sequences, reactivities, masks in cv_loader:
                sequences, reactivities, masks = sequences.to(device), reactivities.to(device), masks.to(device)
                predictions = model(sequences)
                
                # 實作 Kaggle Clipped MAE 評估標準
                preds_clipped = torch.clamp(predictions, 0.0, 1.0)
                reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
                loss_matrix = torch.abs(preds_clipped - reacts_clipped)
                masked_loss = loss_matrix * masks
                
                actual_nucleotides_count = masks.sum()
                if actual_nucleotides_count > 0:
                    cv_loss = masked_loss.sum() / actual_nucleotides_count
                else:
                    cv_loss = torch.tensor(0.0).to(device)
                    
                total_cv_loss += cv_loss.item()
                
        avg_cv_loss = total_cv_loss / len(cv_loader) if len(cv_loader) > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | CV Loss: {avg_cv_loss:.4f}")
        
        if avg_cv_loss < best_cv_loss:
            best_cv_loss = avg_cv_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "rna_reactivity_model_weights_v2.pth")
            print(f"  🌟 CV Loss 創下新低 ({best_cv_loss:.4f})！已儲存最佳權重。")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ CV Loss 未能降低 (連續 {epochs_no_improve}/{early_stop_patience} 個 epoch)")
            if epochs_no_improve >= early_stop_patience:
                print("  🛑 啟動 Early Stopping，提早結束訓練以防止 Overfitting！")
                break
            
        # 根據 CV Loss 更新 Scheduler，若連續多次沒有新低，則降低 Learning Rate
        scheduler.step(avg_cv_loss)
        
    print("[SUCCESS] 訓練結束！最佳模型權重已儲存為 rna_reactivity_model_weights_v2.pth")
    
    print("\n=== 正在進行 Test Set 最終評估 ===")
    model.load_state_dict(torch.load("rna_reactivity_model_weights_v2.pth"))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for sequences, reactivities, masks in test_loader:
            sequences, reactivities, masks = sequences.to(device), reactivities.to(device), masks.to(device)
            predictions = model(sequences)
            # 實作 Kaggle Clipped MAE 評估標準
            preds_clipped = torch.clamp(predictions, 0.0, 1.0)
            reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
            loss_matrix = torch.abs(preds_clipped - reacts_clipped)
            masked_loss = loss_matrix * masks
            actual_nucleotides_count = masks.sum()
            if actual_nucleotides_count > 0:
                test_loss = masked_loss.sum() / actual_nucleotides_count
            else:
                test_loss = torch.tensor(0.0).to(device)
            total_test_loss += test_loss.item()
            
    avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    print(f"🎯 最終 Test Loss: {avg_test_loss:.4f} (此為模型真實預測能力的客觀指標)")

if __name__ == "__main__":
    train_model()
