import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# =====================================================================
# 教學筆記 (Andrew Ng 課程概念對應):
# 1. Feature Engineering (特徵工程): 
#    神經網路無法直接理解 'A', 'C', 'G', 'U' 這樣的字串。
#    我們必須將它們轉換為數字。最常用的方法是 One-Hot Encoding (獨熱編碼)，
#    例如 A = [1,0,0,0], C = [0,1,0,0]。
# 2. Vectorization (向量化):
#    為了發揮 GPU 的平行運算優勢，我們將所有資料打包成形狀固定的矩陣 (Tensor)。
#    即使 RNA 長度不同，我們也會用補零 (Zero Padding) 的方式讓它們長度一致。
# =====================================================================

class RNA3DDataset(Dataset):
    """
    自定義的 PyTorch Dataset，專門用來讀取序列與 3D 座標。
    """
    def __init__(self, sequences_csv, labels_csv=None, max_length=200):
        self.max_length = max_length
        self.char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        
        # 嘗試讀取真實的 Kaggle CSV
        if os.path.exists(sequences_csv) and (labels_csv is None or os.path.exists(labels_csv)):
            print(f"載入真實資料: {sequences_csv}")
            self.seq_df = pd.read_csv(sequences_csv)
            if labels_csv:
                self.label_df = pd.read_csv(labels_csv)
            else:
                self.label_df = None
            self.mock_data = False
            self.num_samples = len(self.seq_df)
        else:
            print("[警告] 找不到真實的 CSV 檔案。")
            print("為了讓您能先測試與驗證神經網路，我們自動生成 1000 筆模擬的 RNA 3D 資料！")
            self.mock_data = True
            self.num_samples = 1000

    def __len__(self):
        # 告訴 DataLoader 這個資料集總共有多少筆資料 (Andrew Ng 課程中的 m 值)
        return self.num_samples

    def __getitem__(self, idx):
        """
        當 DataLoader 準備好要抓取一個 Batch 時，會重複呼叫這個函數來取得單筆資料。
        """
        if self.mock_data:
            # --- 模擬資料生成 ---
            # 隨機產生一段長度介於 50 到 max_length 的序列
            length = np.random.randint(50, self.max_length)
            seq_str = ''.join(np.random.choice(['A', 'C', 'G', 'U'], size=length))
            
            # 模擬的 3D 座標 (x, y, z)，形狀為 (length, 3)
            # 這裡我們隨機生成一些空間座標來模擬原子的位置
            coords = np.random.randn(length, 3).astype(np.float32)
        else:
            # --- 真實資料讀取 (根據 Kaggle Stanford RNA 3D 格式) ---
            # (此處假設真實資料的讀取邏輯，可能需要根據下載後的實際 CSV 欄位名稱微調)
            seq_str = self.seq_df.iloc[idx]['sequence']
            length = len(seq_str)
            
            if self.label_df is not None:
                # 尋找對應這個 sequence 的所有 3D 座標
                target_id = self.seq_df.iloc[idx]['target_id']
                target_labels = self.label_df[self.label_df['ID'].str.startswith(target_id)]
                # 提取 x, y, z 欄位
                coords = target_labels[['x', 'y', 'z']].values.astype(np.float32)
            else:
                coords = np.zeros((length, 3), dtype=np.float32)

        # 限制最大長度 (截斷)
        seq_str = seq_str[:self.max_length]
        coords = coords[:self.max_length]
        actual_length = len(seq_str)

        # ==========================================
        # 1. 將 RNA 序列轉換為 One-Hot Matrix
        #    矩陣形狀: (max_length, 4)
        # ==========================================
        one_hot = np.zeros((self.max_length, 4), dtype=np.float32)
        for i, char in enumerate(seq_str):
            if char in self.char_map:
                one_hot[i, self.char_map[char]] = 1.0

        # ==========================================
        # 2. 將 3D 座標轉換為固定大小的 Matrix
        #    矩陣形狀: (max_length, 3)
        #    不足 max_length 的部分我們補零 (Zero Padding)
        # ==========================================
        padded_coords = np.zeros((self.max_length, 3), dtype=np.float32)
        padded_coords[:actual_length] = coords
        
        # ==========================================
        # 3. 建立 Padding Mask (遮罩)
        #    這非常重要！我們必須告訴神經網路哪些部分是「真實的核苷酸」，
        #    哪些部分是我們為了對齊而「補零」的。1 代表真實，0 代表補零。
        #    在計算 Loss 時，我們會忽略補零的部分。
        # ==========================================
        mask = np.zeros((self.max_length,), dtype=np.float32)
        mask[:actual_length] = 1.0

        # 將 numpy 陣列轉換為 PyTorch Tensor
        return torch.tensor(one_hot), torch.tensor(padded_coords), torch.tensor(mask)

if __name__ == "__main__":
    # 簡單的測試程式碼：確保我們的 Dataset 與 DataLoader 能正常運作
    print("初始化 RNA 3D Dataset...")
    dataset = RNA3DDataset(sequences_csv="dataset/train_sequences.csv", labels_csv="dataset/train_labels.csv")
    
    # DataLoader 會負責把單筆資料打包成批次 (Batches)
    # batch_size=32 代表一次處理 32 條 RNA (Andrew Ng 課程中的 Mini-batch Gradient Descent)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 拿出一批資料看看
    for sequences, coordinates, masks in dataloader:
        print("\n成功取出一個 Batch！")
        print(f"Sequences shape (Batch, Max_Len, 4): {sequences.shape}")
        print(f"Coordinates shape (Batch, Max_Len, 3): {coordinates.shape}")
        print(f"Masks shape (Batch, Max_Len): {masks.shape}")
        break
