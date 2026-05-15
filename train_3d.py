import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 引入我們剛剛寫好的模組
from data_pipeline_3d import RNA3DDataset
from model_3d import RNAPredictor3D

# =====================================================================
# 教學筆記 (Andrew Ng 課程概念對應):
# 1. Forward Propagation (前向傳播): 
#    資料進入神經網路，經過 CNN 和 LSTM，最後吐出預測的座標。
# 2. Loss Function (損失函數): 
#    我們使用 MSELoss (Mean Squared Error)。
#    它會計算「預測的座標」與「真實實驗座標」之間的直線距離平方和。
#    我們的目標是讓這個 Loss 越小越好。
# 3. Backpropagation (反向傳播) & Optimizer:
#    根據 Loss，計算每一個神經元權重 (Weight) 對誤差的貢獻程度 (Gradients)。
#    然後由 Optimizer (我們用 Adam) 依照 Learning Rate (學習率) 去更新權重。
# =====================================================================

def train_model():
    print("=== 開始準備 RNA 3D 結構訓練流程 ===")
    
    # ---------------------------------------------------------
    # 步驟 1: 準備 Dataset 與 DataLoader
    # ---------------------------------------------------------
    # 如果找不到真實 Kaggle 資料，這裡會自動生成 1000 筆模擬資料供測試
    dataset = RNA3DDataset(sequences_csv="dataset/train_sequences.csv", 
                           labels_csv="dataset/train_labels.csv", 
                           max_length=200)
    
    # batch_size=32 意味著每處理 32 條 RNA 才更新一次神經網路的權重 (Mini-batch)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # ---------------------------------------------------------
    # 步驟 2: 初始化模型、Loss Function 與 Optimizer
    # ---------------------------------------------------------
    # 檢查電腦是否有 GPU，有的話就把模型放上 GPU 提升訓練速度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的訓練設備: {device}")
    
    model = RNAPredictor3D().to(device)
    
    # MSELoss: 均方誤差 (Andrew Ng 課程中提到的 Linear Regression Cost Function)
    criterion = nn.MSELoss(reduction='none') 
    # 注意: 我們設定 reduction='none' 是因為我們需要手動用 Mask 過濾掉補零的地方
    
    # Adam: 比傳統 Gradient Descent 更聰明的方法，會自動調整每個參數的 Learning Rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ---------------------------------------------------------
    # 步驟 3: 開始 Epoch 迴圈 (Training Loop)
    # ---------------------------------------------------------
    num_epochs = 5 # 這裡為了測試先設 5 次，真正的比賽通常會跑 50~100 次以上
    
    for epoch in range(num_epochs):
        model.train() # 將模型設定為訓練模式
        total_loss = 0.0
        
        # 從 DataLoader 裡面抓取一個個 Batch
        for batch_idx, (sequences, coords, masks) in enumerate(dataloader):
            # 將資料放進正確的設備 (GPU or CPU)
            sequences = sequences.to(device)
            coords = coords.to(device)
            masks = masks.to(device)
            
            # 1. 將舊的 Gradient 清零 (PyTorch 規定動作)
            optimizer.zero_grad()
            
            # 2. Forward Propagation (預測)
            predictions = model(sequences) # 形狀: (Batch, Max_Len, 3)
            
            # 3. 計算 Loss
            # 先算出每一個核苷酸的 3D 距離誤差
            loss_matrix = criterion(predictions, coords) # 形狀: (Batch, Max_Len, 3)
            
            # 將 x, y, z 三個維度的誤差加總
            loss_per_nucleotide = loss_matrix.sum(dim=2) # 形狀: (Batch, Max_Len)
            
            # 套用 Mask (遮罩)! 
            # 把我們為了補齊長度而補零 (Zero Padding) 的地方的誤差直接歸零。
            masked_loss = loss_per_nucleotide * masks
            
            # 計算這個 Batch 的平均誤差 (只平均「真實長度」的總和)
            actual_nucleotides_count = masks.sum()
            if actual_nucleotides_count > 0:
                final_loss = masked_loss.sum() / actual_nucleotides_count
            else:
                final_loss = torch.tensor(0.0, requires_grad=True).to(device)
            
            # 4. Backpropagation (反向傳播計算梯度)
            final_loss.backward()
            
            # 5. 更新神經網路權重 (Gradient Descent)
            optimizer.step()
            
            total_loss += final_loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
        
    print("\n[SUCCESS] 訓練測試完成！模型具備正常的學習與收斂能力。")

if __name__ == "__main__":
    train_model()
