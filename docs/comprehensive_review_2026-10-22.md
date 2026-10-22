# RNA reactivity prediction：完整審查與程式解說

審查日期：2026 年 10 月 22 日（Asia/Taipei）
專案：`rna-reactivity-prediction`；分支：`codex/exact-length-results`
起始版本：`df2d87bd98ce8ad5b4e27766a98ad73cd5085ada`；起始工作區乾淨。

## 1. 審查結論與使用範圍

**這個專案已具備可核查的探索性研究與工程實作證據，可以作為 PhD 申請的研究經驗；尚不能宣稱已完成獨立最終測試或證明跨 RNA 家族泛化。** 本次直接修正已確認的程式與文件問題，保留 V17／V18 的原始數值和封存圖片，沒有為取得較佳分數變更 cohort、切分或模型，也沒有啟動新的 Kaggle GPU run、推送或合併 main。

V18 有 26 個 checksum-listed compact artifacts，比 V17 少兩個是因為未訓練 sequence-only Transformer，故沒有對應的 `vienna_comparison.csv`／`vienna_aggregate.csv`；real-vs-zero／shuffled 仍在 paired bootstrap 表中。

V18 現在是 **COMPLETE**，API 明確回傳版本 18；不是依交接訊息推定。36 組模型／seed／切分的已存 checkpoint 均已用重新下載的官方 current CSV 在 CPU 重播 CV。最大 GPU／CPU MAE 差異為 **1.4496×10⁻⁸**。V17 的 54 組與 V18 的 36 組 weighted／macro 指標及全部 90 列 bootstrap 結果亦已獨立重算。

本報告中的「CV」指開發用保留驗證分區，**不是 k-fold cross-validation**。clustered 的三個 seed 共用同一個 CV；random／grouped 則同時改變切分與訓練亂數。不能把三個 seed 當成三個獨立生物樣本。

查核材料包括 [逐檔清單](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/file_inventory.csv)、[函式／類別索引](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/function_inventory.csv)、[獨立指標驗證](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/metric_verification.json)、[checkpoint 重播明細](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/checkpoint_replay.csv)。清單逐檔列出 SHA、行數、用途與驗證類別；「程式碼已閱讀」不等於「每條程式分支均在真實硬體執行」。

## 2. 專案究竟在預測什麼

輸入是一條由 A、C、G、U 組成的 RNA 序列，輸出是每個核苷酸位置在 `2A3_MaP` 與 `DMS_MaP` 兩種實驗中的化學反應性。這是連續數值的回歸問題，反應性與局部結構環境有關，但**不等同於直接求出 RNA 三維座標、鹼基配對真值或生物功能**。競賽的資料定義可見 [Kaggle 官方資料說明](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data)。

研究問題是：在相同資料與評估規則下，局部序列模式、雙向序列上下文、注意力與 ViennaRNA 預測結構各提供多少可觀察的效益？另一問題是：允許相同 RNA 跨訓練與驗證分區的 random split，與 exact-grouped、similarity-clustered 結果有何差異？分數差同時受到切分組成影響，不能直接視為資料洩漏的因果效應。

## 3. 完整資料流程

```text
官方 current train_data.csv
  → 完整檔案 SHA-256
  → SN_filter == 1 的合格 rows
  → seed 42 抽取 1,000 條 unique sequence
  → 第二次掃描保留這些 RNA 的所有合格 profiles
  → 驗證 ACGU、assay、連續 target 欄位、有限值及長度
  → 建立 4D 序列／7D 結構與控制組 Dataset
  → random／exact-grouped／固定 MMseqs manifest
  → 每個 train 分區擬合 assay mean／median baseline
  → 初始化模型 → 訓練 → 每 epoch CV → 儲存最佳 checkpoint
  → 重新載入 checkpoint 評估 CV 與原始預測健康度
  → 先合併同一 RNA profiles，再計算 weighted／macro 指標
  → 配對 cluster bootstrap、分層誤差、圖表、checksum
  → compact archive；獨立 final test 入口目前鎖定
```

`SN_filter` 只決定 row 的納入資格，並不是模型輸入。測量缺值也不應變成 0 標籤：Dataset 內雖以 0 填補數值位置，但 mask 為 0，loss 與分母不計入。另一個 assay 未量測的輸出通道也同樣不計分。

### 已重新取得並驗證的資料

| 項目 | 實際結果 |
|---|---:|
| 官方 CSV 大小 | 2,364,840,039 bytes |
| CSV rows | 1,643,680 |
| SN_filter 合格 rows | 438,239 |
| 合格 unique sequences | 243,009 |
| V17／V18 抽樣 cohort | 1,000 sequences、1,781 profiles |
| assay profiles | 2A3：845；DMS：936 |
| 同時具有兩種 assay 的 RNA | 682 |
| 僅一種 assay 的 RNA | 318 |
| 有效 measured targets | 176,351 |
| target 矩陣中的 NaN | 190,535；包含未量測及超出長度的欄位 |
| 完全重複 rows | 0 |
| 重複 sequence／assay keys | 99；保留，可能是不同實驗 profiles |
| 小於 0／大於 1 的 targets | 27,350／19,971 |
| infinity／超出 RNA 長度的有限 targets | 0／0 |
| 長度範圍與中位數 | 115–206 nt；177 nt |

兩種 cohort checksum 使用不同定義：run manifest 的 `489cab…` 是排序後的序列字串，similarity manifest 的 `9093b9…` 是排序後的 sequence digests，因此不應要求兩者字面相等；兩者已對回相同成員集合。assignment SHA 為 `d63a4c523fd2b68169ef4c45b22b93132cf78ca7a45aab3bae16067136a60198`。

完整 source SHA 為 `527fbcd94ea474054a08346df356ed29381cf2ecb41a84149d8f62c4865a1684`；cohort 含量測與 metadata 的 SHA 為 `b22fa846f25f22bb096045abb9ee5706028e1f60d78cf751c6329bfb7410a89b`。新下載檔案、V17 與 V18 一致。獨立 metadata 掃描使用不同 chunk size，仍選出完全相同的 1,000 條序列。證據：[source_verification.json](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/source_verification.json)。

抽樣不是「前 1,000 rows」；先對合格序列去重，再隨機抽序列，第二次掃描取回全部合格 assay rows。因此可以保留分散在檔案不同位置的兩種實驗。抽樣順序依來源首次出現順序，重現時必須保持來源 SHA、seed 與 Python 隨機抽樣行為，而非只提供同樣的序列集合。

原始超出 [0,1] 的數值留在資料中，計算訓練 target 與評估時才裁切。沒有合成研究資料替代路徑。測試中的小型 fixtures 僅驗證工程邏輯；本次真實資料兩 epoch smoke 亦明確標記為工程測試，未加入研究分數。

### 記憶體與快取

loader 只讀 metadata 與 `reactivity_數字`，排除未使用的 `reactivity_error_*`，targets 為 float32。V18 cohort DataFrame 約 2,335,588 bytes；每份 targets 矩陣為 1,467,544 bytes。四個 Dataset 各自保留 targets／篩選後 frame，並非完全共享零拷貝。ViennaRNA 結構先按 unique sequence 計算一次，再供 shuffled／zero 使用；one-hot 特徵仍在 `__getitem__` 每次建立，沒有磁碟特徵快取。

chunked parsing 降低暫存需求，**不表示 `--max-sequences 0` 的全體 Dataset 記憶體固定不變**。完整合格 cohort 的單份 206 欄 float32 targets 約 361 MB，還要加上多份 Dataset、DataFrame、worker 與中間陣列。本次沒有跑全體 cohort 訓練或峰值 RAM benchmark，不能把 1,000-sequence 成功外推為全資料規模已驗證。

## 4. 主要模組與函式如何合作

| 模組與入口 | 意義、輸入／輸出與關係 |
|---|---|
| [src/data_loading.py:15](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/data_loading.py:15) | 找官方 current CSV，拒絕 OLD 路徑與歧義來源；`load_unique_sequence_subset` 執行兩階段取樣。研究 runner 沒有啟用 legacy opt-in。 |
| [src/data_pipeline_reactivity.py:34](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/data_pipeline_reactivity.py:34) | 把一筆 profile 轉成 features、targets、mask 三個 tensor；assay 決定 target 放在哪個通道。驗證連續 target 欄位，現在拒絕默默截斷、無有效值及 infinity。 |
| [src/data_pipeline_reactivity.py:29](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/data_pipeline_reactivity.py:29) | 只把序列與記錄下來的 ViennaRNA model details 傳給 MFE；不輸入反應性、SN_filter 或 CV 分數。 |
| [src/model_reactivity.py:36](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/model_reactivity.py:36) | 卷積後以每個位置自己的通道作 LayerNorm，再把 padding activations 歸零，避免下一層卷積從尾端吸收假的 padding 訊息。 |
| [src/model_reactivity.py:23](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/model_reactivity.py:23) | 由序列 mask 求真實長度，pack 後做雙向 LSTM，再用固定 `total_length` 還原。 |
| [src/model_reactivity.py:188](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/model_reactivity.py:188) | CNN→Bi-LSTM→Transformer→residual→線性兩通道輸出；4D／7D 由同一實作支援。 |
| [src/splitting.py:6](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/splitting.py:6) | unique sequence 作分組單位；random 版本則直接切 rows；`validate_split_integrity` 查重複、缺漏與越界索引。 |
| [src/similarity_splitting.py:126](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/similarity_splitting.py:126) | 依有效 target 數分配完整 cluster，使用確定性的排序與 tie-break；`validate_similarity_manifest` 查 hash、cluster 成員、來源與六方向 audit。 |
| [src/mmseqs_clustering.py:376](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/mmseqs_clustering.py:376) | 寫入 SHA 標記且排序的 FASTA、呼叫 MMseqs clustering、轉成穩定 cluster ID、建立 manifest。 |
| [src/mmseqs_clustering.py:584](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/mmseqs_clustering.py:584) | 搜尋漏邊→合併完整 components→重新分配→從頭重做全部六方向，直到通過或失敗。 |
| [experiments/build_similarity_manifest.py:34](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/build_similarity_manifest.py:34) | 從真實 targets 的有限值數建立 sequence weights，再呼叫 clustering／repair；allocation 使用量測可用性，不使用誤差或反應性大小。 |
| [experiments/run_ablation.py:800](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:800) | 讀 config、資料與 manifest，建立 Dataset、baseline、模型 registry 與所有 seed／split 訓練；統一寫出結果。 |
| [experiments/run_ablation.py:478](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:478) | Adam、masked L1、ReduceLROnPlateau、CV checkpoint 與 early stopping 的執行位置。 |
| [experiments/run_ablation.py:618](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:618) | 重新載入權重，按 measured mask 聚合誤差、原始預測統計與結構分層；以 row_indices 對回原 profile。 |
| [experiments/run_ablation.py:204](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:204) | 只看 train indices，先裁切有效 targets，依 assay 算 mean；median helper 共用取值規則。 |
| [src/prediction_health.py:7](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/prediction_health.py:7) | 以 profile 的充分統計量合併 raw mean／std、範圍、clipping 比率與 baseline improvement。 |
| [src/evaluation_statistics.py:111](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/evaluation_statistics.py:111) | 同一 seed／split／partition／variant 內，先合併同一 RNA 的所有 profiles。 |
| [src/evaluation_statistics.py:337](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/evaluation_statistics.py:337) | 嚴格配對 sequence／cluster／seed／分母；每次抽完整 cluster，分別計算 weighted 與 macro 改善區間。 |
| [experiments/validate_development_archive.py:17](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/validate_development_archive.py:17) | V17／V18 compact evidence 的獨立一致性入口；已補強 convergence、baseline、health、membership、選模與 bootstrap 檢查。 |
| [experiments/validate_archive.py:241](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/validate_archive.py:241) | 專用 V9／V10 的歷史 P100、1,820-profile schema 驗證器；不可誤用來判定 T4 開發 run。 |
| [experiments/archive_kaggle_development.py:13](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/archive_kaggle_development.py:13) | 檢查 COMPLETE、遍歷分頁、核對 commit 與逐檔 checksum，只下載 compact artifacts。已實際下載驗證。 |
| [kaggle/gpu_ablation.py:248](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/kaggle/gpu_ablation.py:248) | Kaggle T4×2 preflight、相依套件、精確 commit checkout、manifest 與訓練啟動，最後檢查完整結果格。 |
| [kaggle/submit_gpu.py:12](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/kaggle/submit_gpu.py:12) | 將指定 SHA 注入暫存 runner；smoke／development／convergence 三種提交設定；原始 placeholder 留在 repo。 |
| `kaggle/run_gpu_ablation.sh` | 舊版直接提交 placeholder，現在統一轉交 `submit_gpu.py`，要求明確 commit 與 mode。 |
| `kaggle/download_data.sh` | 資料下載入口；會安裝 Kaggle 用戶端並下載競賽內容。本次採既有 CLI 只下載 current CSV，沒有額外執行全量下載腳本。 |
| `configs/ablation_config.yaml` | batch=64、lr=0.001、weight decay=1e-5、scheduler patience=2／factor=0.5；其 15 epochs 是預設值，V17／V18 分別由 CLI 覆寫為 20／60。 |
| `tests/`、`.github/workflows/test.yml` | 工程 fixtures、模型梯度／padding、資料、統計、archive 與回歸測試；CI 用 CPU Torch 2.7.1。本次本機是 Torch 2.10.0，沒有聲稱重跑 GitHub CI。 |
| `README.md`、`docs/`、`experiments/experiment_notes.md` | 現行入口說明、final protocol 草案及歷史學習紀錄；早期機制推論不等同受控研究結果，已修正過強措辭。 |
| `results/` | V9／V10 為 OLD-data 歷史證據；V17 為六模型短訓練開發；V18 為四模型收斂追蹤。 |

`RNAReactivityCNN_LSTM_Transformer` 與 `RNAReactivityPredictor(input_dim=4)` 的架構實作重複；前者仍被模型測試引用，現行 registry 使用後者，屬可清理的重複維護點，不是本次證實的數值錯誤。保留相容介面，未擅自刪除。`src/__init__.py`、`tests/__init__.py` 為套件標記；LICENSE、mailmap、gitignore 為專案 metadata。未追蹤的空 notebooks 目錄不是有效執行入口。

## 5. 模型、mask 與訓練細節

### 輸入與輸出

單筆 features 為 `[206,4]` 或 `[206,7]`。前四個通道是 ACGU one-hot；另三個為 dot-bracket `(`、`)`、`.`，分別表示與下游位置配對、與上游位置配對、未配對。它們**不直接編碼配對對象的索引**。targets 與 mask 均為 `[206,2]`，通道順序為 2A3、DMS。mini-batch 的形狀在前面多一個 B。

CNN 為兩層 kernel=5，通道 64→128；兩層之後的局部感受野為 9 個位置。Bi-LSTM 有兩層，每方向 hidden=128，合計每位置 256 維。Transformer 有兩層、8 heads、feed-forward=1024；使用 Bi-LSTM 上下文而未另加顯式 positional encoding。輸出為線性層的兩個無界數值，並無 sigmoid。Dropout=0.3。

| Variant | 參數與解讀 |
|---|---|
| CNN only | 43,074；局部模式 reference，不是性能理論下界 |
| CNN + Bi-LSTM | 702,786 |
| Transformer 4D | 2,282,306 |
| real／zero／shuffled 7D | 各 2,283,266；同 seed 同架構初始化；差異只在結構輸入 |

zero 控制的三個結構通道全零；shuffled 使用 sequence 與固定 control seed 314159 的 SHA 派生亂數，只打亂結構通道的位置，不改序列與 targets。這是每條 RNA 一次固定的控制排列，不涵蓋多種排列的不確定性。real 與 zero／shuffled 的比較比「real 與不同大小的 Bi-LSTM」更適合檢驗這些結構特徵的增量效益。

### padding 與雙 GPU

padding mask 從前四個 sequence channels 的和是否為 0 得出，與 target missingness 分開。LayerNorm 不跨 batch 或序列長度計算；每層 CNN 後歸零 padding。packed LSTM 不讀尾端填補；unpack 指定原始完整長度，使不同 GPU 子批次仍回傳相同 L。Transformer 的 `src_key_padding_mask` 排除 padding keys；padding queries 可能仍有數值，但不會成為其他位置可用的 keys，也不進 loss。

DataParallel 切的是 batch 維度。所有子 GPU 的完整預測先 gather，才在主 GPU 用 **總 absolute error／總有效位置數** 計算 loss，並未平均兩張卡各自的 MAE。因此不同 assay、不同缺值量與最後不足整批的 batch 不會因兩個 GPU 分母不相等而被錯誤等權。`model.module.state_dict()` 避免把 `module.` 前綴寫進一般模型 checkpoint。PyTorch 官方亦說明 packed RNN 在 DataParallel 中需要固定 unpack 長度：[官方 FAQ](https://docs.pytorch.org/docs/stable/notes/faq.html)。

證據層次：原始 V18 environment 記錄兩張 Tesla T4、DataParallel=true，log 有 36 次 DataParallel 訓練訊息；本次讀程式確認 gather 與分母位置，執行真實 autograd 的不等分母分片回歸測試，並重播全部 checkpoint。**本機 CUDA unavailable、GPU count=0，因此沒有重新執行 T4×2 硬體測試或比較單卡／雙卡完整訓練軌跡**。不能以 CPU 分片測試冒充雙卡測試。

### loss、最佳模型與停止條件

訓練 loss：`sum(mask * abs(raw_prediction - clip(target,0,1))) / sum(mask)`。評估則同時裁切 prediction 與 target。這讓超出範圍的訓練輸出仍有 L1 梯度；兩種函數不是完全相同的最佳化目標。optimizer 是 Adam 搭配 weight decay，並非 AdamW。scheduler 使用 CV plateau 降 learning rate。

每次出現任何更小的 CV，都儲存最佳 checkpoint；early stopping 只有改善超過 0.0001 才重設 patience。故最佳 epoch 可能距離停止 epoch 少於 8，這不是 checkpoint 遺失。最佳 checkpoint 重載後現在會檢查 CV 與選中 epoch 是否一致。checkpoint 只含模型權重，沒有 optimizer／scheduler／RNG state；它支援推論重播，**不支援無縫接續中斷訓練**。

Torch、NumPy、Python random 現已統一設 seed；sample seed、structure-control seed 與 training seed 角色分開。deterministic algorithms 使用 warn_only，CUDA 仍可能有非位元級確定性。Kaggle 使用既有 Torch 2.10.0+cu128，requirements／CI 的 2.7.1 不等同當次 GPU runtime，必須以 environment.json 和已保存的 image digest 重現環境。

## 6. ViennaRNA 與切分可支持的結論

V17／V18 使用 ViennaRNA 2.7.2 的 `RNA.fold(sequence)`，沒有反應性 constraints。修正後改用明確 model-details 物件的 `fold_compound(..., OPTION_MFE).mfe()`，新 run 會記錄全部可序列化的 model-details 欄位及結構集合 SHA。**在這次真實 1,000-sequence cohort 上，新舊呼叫產生的結構逐條完全相同**。既有封存 environment 沒有這些完整欄位，不能把本次新紀錄冒稱當時原始紀錄。

本次實際 settings：temperature=37°C、dangles=2、noLP=0、noGU=0、noGUclosure=0、circ=0、gquad=0、min_loop_size=3、max_bp_span=-1、salt=1.021；其餘 scalar settings 完整保存在 [checkpoint_verification.json](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/checkpoint_verification.json)。沒有使用 probing 值、ensemble BPP、外加 constraint 或 pseudoknot；單一 MFE 不描述 ensemble uncertainty。[ViennaRNA 官方 model-details 說明](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/grammar/model_details.html)。

random split 切 rows，重複 RNA 可以跨分區；exact-grouped 保留完整同序列 profiles；clustered 則保留整個 MMseqs component。70／15／15 對 clustered 指的是有效 target 權重，**不是要求每區序列數正好 700／150／150**。

V17／V18 frozen split 有 921 clusters：train／CV／test 為 751／128／121 條 RNA、1,234／274／273 profiles、123,403／26,492／26,456 個有效位置。CV 包含 86 clusters。兩次漏邊合併已記錄在 manifest；完整 cohort 的有效位置數與三區權重總和均為 176,351。

稽核使用 nucleotide `search-type=3`、forward strand=1、k=8、identity=0.8、coverage=0.8、cov-mode=0、seq-id-mode=0、alignment-mode=3、sensitivity=7.5、mask=0、threads=1。identity 的分母是 alignment length；coverage mode 0 要求 query 與 target coverage，操作上對兩序列均足夠覆蓋。依據為 [MMseqs2 官方參數定義](https://github.com/soedinglab/MMseqs2/blob/master/src/commons/Parameters.cpp) 與 [官方指南](https://mmseqs.com/latest/userguide.pdf)。

本次以實際序列重做 train→CV、CV→train、train→test、test→train、CV→test、test→CV 六方向，全部零 qualifying hits。這是 sequence-only 稽核，未推論或評估 test targets。重驗版本為 18-8cc5c+ds-1，原始版本為 13-45111+ds-2：因此是**新版本對既有切分的獨立稽核**，不是原始 clustering 的位元級重建。詳細證據：[similarity_reaudit.json](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/similarity_reaudit.json)。

只能說「在記錄的序列搜尋方式與門檻下，未偵測到跨分區符合門檻的命中」。搜尋仍有 heuristic、strand、局部對齊、低於門檻關係與家族分類限制，不能宣稱普遍 homology-free。

## 7. V18 的結果與統計意義

| 模型／baseline | clustered CV weighted MAE | 三個訓練 seeds 的 sample SD |
|---|---:|---:|
| Bi-LSTM | 0.220151 | 0.001382 |
| real ViennaRNA | **0.204259** | 0.000647 |
| zero channels | 0.228237 | 0.002627 |
| position-shuffled | 0.230184 | 0.000237 |
| train-only assay mean | 0.293335 | 固定訓練分區，常數相同 |
| train-only assay median | 0.279594 | 固定訓練分區，常數相同 |

real 模型的 macro-sequence MAE 為 **0.208889 ± 0.000222**。V17 real clustered weighted MAE 約 0.2086；V18 較低，但這是同一開發資料上延長訓練並重新選 epoch 的觀察，不能把改善當成新的獨立確認實驗。

weighted MAE 先加總所有有效核苷酸誤差再除以有效位置數；多 profiles 或長序列因此較有權重。macro 指標先將同一 RNA 的 profiles 誤差與分母合併，再算每條 RNA 的 MAE，最後等權平均；不是先平均 assay scores，也不是每筆 profile 等權。全域 assay 比重依量測數決定，並非強制 2A3、DMS 各占一半。

mean／median baseline 只使用 train 的有效 clipped targets，各 assay 一個常數；median 對 L1 是必要的強基準。九個 seed／split 組的 mean、median 常數與 CV 分數均從真實資料以獨立 NumPy 寫法重算一致。分母與模型完全一致；clustered baseline 每個 seed 重複相同數值是預期行為，不是三個獨立估計。

bootstrap 對 reference 與 candidate 嚴格配對，再抽整個 similarity cluster；同一個 cluster 內的 RNA 與 profiles 一起移動。每個 seed 各做 10,000 次，不將 seeds 堆疊為額外樣本。real 相對 zero 的三個 weighted 改善 95% 區間約為 [0.0143,0.0275]、[0.0171,0.0327]、[0.0180,0.0323]；相對 shuffled 亦全部為正。這些是**對已選 checkpoint 與開發 CV 的條件式探索區間**，沒有校正反覆選模型、選 epoch、多重比較與研究決策帶來的偏差。`probability_improvement=1.0` 代表抽樣中全部改善，不是生物學命題為真的後驗機率。

### 收斂與預測健康度

V18 共 1,808 個 epoch records；25／36 組觸發 early stopping，11 組到達 60 epochs 而未觸發。clustered real 三 seeds 的 best epochs 為 33、35、48，最後 epochs 為 41、43、49。只能說停止規則已如實執行，不能證明所有模型達到最佳解或已充分收斂。

36 組選中 checkpoint 均沒有 95% clipping 飽和或近常數 warning；raw prediction SD 約 0.230–0.326、clipped fraction 約 0.10%–6.67%。clustered real 的 raw MAE 約 0.205724，clipped MAE 約 0.204259，clipping fraction 約 3.98%。這些診斷支持模型不是靠全零或大面積裁切取得表面分數，但不證明生物機制。原始逐核苷酸 prediction arrays 並未存入 compact archive；可核查的是原始 profile 充分統計與 checkpoint 重播。[health_verification.json](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/health_verification.json) 記錄獨立重組誤差約 1.1×10⁻¹⁶。

### 圖表審查

檢視四個版本共 11 張既有 PNG 與對應 CSV。V9／V10 為歷史 OLD-data 圖，不作 current 結論；V17 20 epochs 都有三 seeds，平均曲線沒有 early-stopping 組成改變問題，但 18 條線擁擠。V18 原圖在部分 run 停止後繼續平均剩餘 seeds，原標題未充分說明；已另繪下圖，原檔與 checksum 保持不變。新圖以各 split 分面，細虛線為 individual run，叉號為最後 epoch，粗線與 SD 僅在完整三 seeds 皆存在時繪製；沒有向後補值，也沒有把已停止 seed 當作仍在訓練。

![經審查的 V18 訓練曲線](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/v18_training_curves_reviewed.png)

長度分布有 917／1,000 條 RNA 長度為 177 nt。選中 clustered CV 的短序列組僅五條 RNA，而且跨 seed 是相同五條；圖上的 n=5/seed 不能被讀成 15 個不同 RNA。圖表適合作描述，不能支持短 RNA／長 RNA 泛化排名。

## 8. 已確認問題、修正與證據

P1 表示會影響研究解讀或測試紀律；P2 表示會降低可靠性或可重現性。沒有發現需改寫 V17／V18 已封存主要 MAE 的已確認錯誤。

| 編號／嚴重度 | 檔案與根因、影響 | 修正與驗證 |
|---|---|---|
| F01／P1 | [experiments/run_ablation.py:819](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:819)：原本只要求 audited clustered manifest，沒有開發曝露排除或已凍結協定防線，可把開發 cohort 的 test 誤當 final。 | 在資料存取前拒絕 final-eval。真實 subprocess 回歸證明不存在 data／manifest 時仍優先得到鎖定錯誤。協定實作仍為下一階段，不虛構可解鎖狀態。 |
| F02／P1 | [experiments/run_ablation.py:349](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:349)：early stopping 後每 epoch seed 數改變，平均曲線可能反映存活組成。 | 個別曲線＋完整 seed 組平均、三分面、最後 epoch 標記；不等長 history 回歸與實際 V18 圖目視查核。 |
| F03／P2 | [experiments/validate_development_archive.py:73](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/validate_development_archive.py:73)：原 validator 沒有核對 median、initial health、epoch count／stopping、部分 denominators／bootstrap。 | 補齊格子、membership、分母、選模與統計核對；即使重新生成 checksums，五種語意破壞測試仍會被拒絕。 |
| F04／P2 | [src/data_pipeline_reactivity.py:188](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/data_pipeline_reactivity.py:188)：曾以 notna 判定有效，infinity 或只在序列外有值可通過；長 RNA 默默截斷。 | 拒絕 infinity、序列外有限值、無有效量測與超長資料；四種邊界回歸。真實 cohort 沒有這些異常，研究數值不受更動。 |
| F05／P2 | [experiments/run_ablation.py:478](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:478)：全無量測的 CV 使用 max(denominator,1) 可產生假 0 分；非有限 prediction 可能被裁切掩飾。 | 空分母與非有限訓練／CV 明確失敗、重載 checkpoint 比對；empty-CV 不會保存「完美」權重。 |
| F06／P2 | [src/similarity_splitting.py:325](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/similarity_splitting.py:325)：原 validator 未檢查核酸 search_type、strand、kmer。 | 固定驗證 3／1／8，參數破壞測試與真實六方向重驗通過。 |
| F07／P2 | `kaggle/run_gpu_ablation.sh:1`：直接推送未替換 placeholder，與精確 commit 提交流程不一致。 | 統一轉交 submit_gpu.py；shell syntax 通過；沒有為測試腳本而提交新 GPU run。 |
| F08／P2 | [experiments/archive_kaggle_development.py:13](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/archive_kaggle_development.py:13)：遇 pagination 直接失敗，亦未主動要求 COMPLETE。 | 實作頁面遍歷、重複檔／token 拒絕、狀態與 manifest checksum 檢查；以 page_size=20 真實重取 26 檔成功。 |
| F09／P2 | [src/data_pipeline_reactivity.py:29](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/src/data_pipeline_reactivity.py:29)：ViennaRNA 舊 wrapper 的完整 settings／結構 hash 未留在 run environment。 | 新 run 記錄明確 model details 與結構 SHA；全部 1,000 條真實結構新舊一致；不補寫舊檔 provenance。 |
| F10／P2 | [experiments/run_ablation.py:106](/mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction/experiments/run_ablation.py:106) 與 main：Python random 未統一 seed；重複 seed／variant 與非空輸出目錄可能形成混合證據。 | 補 Python seed、拒絕重複設定與非空目錄；完整 suite 與真實 CV 重播無回歸。 |
| F11／P2 | README／results／experiment_notes 的現行與歷史敘述混雜；residual 被描述成保證不降低性能。 | 增加 V18 現況、OLD-data 警示、final gate 與推論限制；移除未受控驗證的機制確言。 |

## 9. 實際驗證清單與邊界

| 實際執行 | 結果 |
|---|---|
| 起始 git status／branch／log、祖先與專案 AGENTS 搜尋 | 工作區乾淨、分支及 SHA 符合；使用者的繁體中文要求沿用 |
| 原始 pytest | 128 passed |
| 修正後 compileall＋完整 pytest | 145 passed；約 12.6 秒的一次完整執行 |
| V9／V10 archive validator | 各 24 組歷史結果通過；重算最大差約 7.42×10⁻⁹ |
| V17／V18 指標獨立實作 | 54＋36 組 weighted／macro；最大 weighted 差約 1.00×10⁻⁸ |
| V17／V18 bootstrap 獨立實作 | 54＋36 列；最大 CI 差約 1.32×10⁻¹⁶ |
| 新下載官方來源＋獨立抽樣 | source 與 cohort hashes、全部 unique sequences 一致 |
| 真實 train-only baselines | 九組分區的 mean／median 常數與 CV 分數一致 |
| V18 全部 checkpoint CPU 重播 | 36／36；最大 CV 差 1.45×10⁻⁸，最大 profile MAE 差 7.63×10⁻⁸ |
| 原始 raw health 充分統計重組 | 36／36；最大差 1.11×10⁻¹⁶ |
| 真實資料訓練 smoke | clustered train 的 12 profiles、CV 的 8 profiles，2 epochs、有限 loss、有 checkpoint；工程用途 |
| 真實 MMseqs sequence-only 搜尋 | 六方向零 qualifying hits；新版本獨立稽核 |
| 修正 downloader 實際 API 下載 | 26 compact 檔 checksum 通過，原始 66 個下載檔另有 inventory |
| 圖表檢視 | 11 張歷史／現行原圖及一張修訂圖；沒有改寫原圖 |
| shell syntax／git diff --check | 通過 |

未執行：新的 T4×2 訓練、單卡／雙卡 trajectory 對照、原 MMseqs 13 的全流程位元重建、全資料 cohort 訓練／峰值 RAM、GitHub 遠端 CI、獨立 final test。沒有將「未執行」改寫成「通過」。本次亦未重播 V17 的原始 checkpoint，V17 的數值驗證以其 compact 誤差分子／分母為基礎。

### 審查覆蓋紀錄

以下分母是明確查核範圍的單位數，不是測試覆蓋率；類別互相重疊，不可合成整體正確率。0／N 表示未觀察到剩餘缺陷，不保證所有未見條件均正確。

| 類別 | 剩餘已觀察缺陷 | 評估 |
|---|---:|---|
| 審查文件與主要流程說明 | 0／8 | 八項使用者工作範圍均有證據或具名限制；獨立 test 是未完成研究階段 |
| 圖表可讀性與敘述 | 1／11 | V18 原 training plot 的存活 seed 平均仍保留作歷史原件；已另附正確版本與警示 |
| dashboard 互動一致性 | N/A | 本交付為程式與研究審查文件，非互動 dashboard |

| 類別 | 剩餘已觀察缺陷 | 評估 |
|---|---:|---|
| current-data 來源與 cohort | 0／2 | V17、V18 source／cohort 一致；當次原始完整 Vienna metadata 缺項已明示 |
| 主要 MAE／macro 計算 | 0／90 | 90 組開發結果獨立重算；不是新增 90 個獨立生物實驗 |
| bootstrap 計算 | 0／90 | 90 列數值重算一致；選擇偏差限制仍存在 |
| baseline／分母規則 | 0／9 | 九組實際 train／CV 分區驗證；無 test baseline 解鎖 |
| current run 跨表一致性 | 0／36 | V18 36 組 checkpoint、profile、sequence、summary、health 對齊 |
| 確認性結論資格 | 1／1 | 尚無獨立 final test，必須保留探索性定位 |

## 10. 如何重現與接續

使用既有 WSL Python：`/home/user/miniconda3/envs/rna_rl_env/bin/python`。本次本機 Python 與套件實際版本在 [local_validation.json](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/local_validation.json) 記錄；Kaggle 的 runtime 應看封存 environment，不可僅照 requirements 宣稱完全相同。

```bash
cd /mnt/c/Users/user/Documents/Codex/2026-08-03/new-chat/work/rna-reactivity-prediction
/home/user/miniconda3/envs/rna_rl_env/bin/python -m compileall -q src experiments kaggle tests
/home/user/miniconda3/envs/rna_rl_env/bin/python -m pytest -q
/home/user/miniconda3/envs/rna_rl_env/bin/python experiments/validate_development_archive.py results/kaggle-v17-9d904ff
/home/user/miniconda3/envs/rna_rl_env/bin/python experiments/validate_development_archive.py results/kaggle-v18-df2d87b
```

本次獨立核驗 scripts 另附於 outputs：`verify_source.py`、`verify_metrics.py`、`verify_checkpoints.py`、`verify_similarity.py`。scripts 明確使用此工作目錄與既有 repo，可重新執行；先保留本任務 `work/current-data/train_data.csv`、`work/current-cohort.pkl`、`work/kaggle-current-raw` 與解包的 MMseqs binary。`verify_checkpoints.py` 最後會另跑兩 epoch 工程 smoke，不產生研究結論。移動任務目錄時須更新 script 中 repo 路徑；它們不是可攜式安裝工具。

完整資料與原始 36 個權重留在本任務 work，compact archive 已放入 repo `results/kaggle-v18-df2d87b`。原始下載 inventory：[raw_download_inventory.csv](/mnt/c/Users/user/Documents/Codex/2026-10-22/rna-astra-comprehensive-review/outputs/raw_download_inventory.csv)。不把來源 CSV、profiles 或模型權重加入 Git。下載時保留 API version metadata，且不得把 log 第一個事件之前約 10,248 秒全部稱為排隊：目前沒有足夠紀錄拆解。可觀察的第一個事件到 outputs ready 約 4,188 秒，其中各模型訓練 elapsed 合計約 3,928 秒。

未來如確定需要新 development GPU run，需先將審查後 commit 推送到已授權分支，再以 `submit_gpu.py --commit <完整已推送 SHA> --mode ...` 提交；這是操作說明，不是本次已執行的動作。不要直接以未釘選的 placeholder runner 提交，也不要用覆寫舊輸出目錄混合兩次實驗。

## 11. PhD 申請可用的表述

可放在 CV 的精簡英文描述：

> Developed and audited a reproducible PyTorch pipeline for RNA chemical-reactivity prediction using measured Ribonanza data, including padding-safe models, sequence-similarity split audits, controlled ViennaRNA feature ablations, and paired cluster-bootstrap evaluation. On a 1,000-sequence development cohort, the structure-augmented model achieved clustered-validation MAE of 0.2043 versus a train-only assay-median baseline of 0.2796; independent final-test evaluation remains pending.

中文說明：建立並檢驗 RNA 化學反應性預測流程，使用真實量測資料、padding-safe 神經網路、相似序列切分稽核、ViennaRNA 特徵控制組與 cluster bootstrap；在 1,000 條序列的開發 cohort 上取得 clustered CV MAE 0.2043，優於 train-only assay median 0.2796，獨立最終測試尚待完成。

應依實際個人貢獻調整「developed」與「audited」，能說明 AI 工具協助了哪些部分，避免把不熟悉或非本人負責的工作寫成獨立完成。學習日誌中的第一人稱時間線是作者自述，本審查不能驗證當時由誰完成或花費多久。

面試時應能回答：為何一個 RNA 有多筆 profiles？為何 random split 會重複序列？為何 training 不裁切 prediction？為何 median 比 mean 更適合 L1 reference？為何三個 seeds 不能當三個獨立資料集？為何 zero／shuffled 控制比只比較兩種不同架構更有解釋力？為何 CV 最佳分數不等於新資料的無偏性能？

不可寫成：完成獨立測試 MAE 0.2043、證明 homology-free、證明 ViennaRNA 發現真實生物機制、預測三維結構、達到競賽領先水準、所有模型充分收斂，或把 V10 OLD-data test 0.1919 當成本次 current-data final result。

## 12. 優先順序與尚待決定事項

1. **先凍結研究問題與選模決策。** 依 clustered CV 討論模型、超參數、seeds、test 大小與是否確認 real-vs-zero／shuffled contrasts；不能從本報告自動推定已批准最終模型。
2. **建立完整開發曝露清冊與新 test cohort。** 最少排除全部 V17／V18 1,000 sequences 及符合門檻鄰居；還要檢查早期 V9／V10 或其他開發是否曝露更多序列。不得僅排除 clustered train／CV。新增 cohort 的選取規則與相似性稽核需記錄來源、版本、hash 與失敗條件。
3. **另行實作並審查 final-eval gate。** 要能檢查已凍結 protocol、曝露清冊、排除證據、固定 model／contrasts 與 test-use history，再考慮解除目前硬鎖；本次沒有實作未獲決定的研究協定。
4. **一次性獨立測試與完整封存。** 預先定義 weighted／macro／assay metrics、cluster 數與 paired uncertainty；test 後若再調參，必須另找新 test。
5. **再處理擴展與維護。** 全 cohort 的共享 targets／特徵快取、確切 runtime pinning、可恢復訓練狀態、重複模型類別與巨型 runner 拆分可作下一批工程工作。沒有證據顯示現在需要為上述整理重新跑昂貴實驗。

這份交付完成的是現有專案的審查、可支持修正與證據重驗；新的確認性研究仍需上述決策與獨立資料，不能由更多相同 CV 重跑取代。
