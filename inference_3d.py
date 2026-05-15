import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 引入我們寫好的模型架構
from model_3d import RNAPredictor3D

# =====================================================================
# 教學筆記 (展示您的 End-to-End 能力):
# 面試時您可以解說：
# 1. Inference (推論): 當模型在 Kaggle 訓練好後，我們只要載入權重檔 (.pth)，
#    就可以不需要 GPU，直接在普通筆電上對全新的序列進行預測。
# 2. Visualization (視覺化): 我們使用 Matplotlib 提取預測出來的 X, Y, Z 座標，
#    並繪製成 3D 空間中的 RNA 骨架。綠色代表 5' 端，紅色代表 3' 端。
# =====================================================================

def sequence_to_onehot(seq_str, max_length=200):
    """將輸入的英文字母序列轉換成神經網路看得懂的 One-Hot Tensor"""
    char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    seq_str = seq_str[:max_length]
    one_hot = np.zeros((max_length, 4), dtype=np.float32)
    for i, char in enumerate(seq_str.upper()):
        if char in char_map:
            one_hot[i, char_map[char]] = 1.0
            
    # 增加一個維度變成 (Batch_Size=1, Max_Len, 4)
    return torch.tensor(one_hot).unsqueeze(0) 

def visualize_rna_3d(sequence, weights_path="rna_3d_model_weights.pth"):
    print("=== 初始化 RNA 3D 推論模型 ===")
    model = RNAPredictor3D()
    
    # 檢查是否已經有從 Kaggle 下載下來的權重檔
    if os.path.exists(weights_path):
        print(f"成功載入訓練好的大腦 (權重檔): {weights_path}")
        # map_location='cpu' 確保在沒有 GPU 的筆電上也能順利載入
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    else:
        print(f"[警告] 找不到權重檔 {weights_path}。")
        print("目前先使用未經訓練的「初始神經元參數」來為您展示視覺化功能！")
        print("（面試前，請務必將 Kaggle 跑完的 .pth 檔案放到這個資料夾內）")

    # 將模型切換為評估模式 (關閉 Dropout 等訓練專用機制)
    model.eval() 

    print(f"\n準備預測的 RNA 序列 (長度 {len(sequence)}): {sequence}")
    input_tensor = sequence_to_onehot(sequence)
    
    # 進行前向傳播 (Forward Pass) 取得預測座標
    with torch.no_grad(): # 告訴 PyTorch 我們現在不是在訓練，不需要計算 Gradient，可以節省大量記憶體
        predictions = model(input_tensor) # 輸出的形狀會是 (1, 200, 3)
    
    # 提取我們輸入長度的有效座標
    seq_length = len(sequence)
    coords = predictions[0, :seq_length, :].numpy()
    
    # 將 x, y, z 分離出來準備畫圖
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    print("\n開始渲染 3D 立體結構圖...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 畫出 RNA 的骨架 (Backbone)
    ax.plot(x, y, z, marker='o', linestyle='-', color='royalblue', markersize=5, alpha=0.8, label="RNA Backbone")
    
    # 標示出起始與結束端點 (面試官最愛看的細節)
    ax.scatter(x[0], y[0], z[0], color='lime', s=150, edgecolor='black', label="5' End (Start)")
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=150, edgecolor='black', label="3' End (Stop)")
    
    ax.set_title("AI Predicted RNA 3D Structure", fontsize=16, fontweight='bold')
    ax.set_xlabel("X Coordinate (Å)")
    ax.set_ylabel("Y Coordinate (Å)")
    ax.set_zlabel("Z Coordinate (Å)")
    ax.legend()
    
    plt.tight_layout()
    
    # 將成果儲存為圖片檔
    save_path = "predicted_rna_3d.png"
    plt.savefig(save_path, dpi=300)
    print(f"✨ 視覺化圖片已成功儲存為: {save_path} ✨")

if __name__ == "__main__":
    # 這是一段供測試用的 RNA 序列 (您可以隨便改成任何您想測試的 mRNA 片段)
    test_sequence = "AUGGCUACGGUCGAAUGCGCUAGCUAGCUAGCUAGCUAGCGGAAUUCCGG"
    visualize_rna_3d(test_sequence)
