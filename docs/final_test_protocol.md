# 獨立測試研究協定 v1

協定 ID：`rna-final-2026-11-08-v1`；凍結日期：2026-11-08（Asia/Taipei）。

**研究設計與選模決策已凍結（DESIGN_FROZEN）；執行仍鎖定（LOCKED_PENDING_COHORT_AND_GATE）。** 本文件取代原草案中的研究決策待定狀態，但不是 test 存取授權。新 cohort 尚未建立、完整開發曝露清冊尚未核定，final-eval 防線尚未實作為可解鎖流程。缺少任一必要證據即停止；不能把 null hash 視為通過。

機器可讀決策、既有證據與三個權重的實際 SHA 見 [凍結清單](research_freeze_2026-11-08.json)；本文件及該清單的位元完整性由 [SHA-256 清單](research_freeze_2026-11-08.sha256) 記錄。Git commit 提供版本歷史；checksum 證明內容一致，不證明研究紀律曾被遵守。

## 1. 凍結的問題與選模理由

唯一確認性問題：**在排除全部開發曝露序列及符合既定相似性門檻鄰居的新 RNA 上，V18 real ViennaRNA 模型的 clipped MAE 是否優於只用既有 train partition 擬合的 assay median baseline？** 同時報告絕對性能與不確定性。

選定 `transformer_vienna_real7`（CNN→Bi-LSTM→Transformer＋三個 MFE 結構通道），採 V18 clustered CV 三個 seeds 的平均 nucleotide-weighted clipped MAE 作唯一選模依據。real=0.2042589064、Bi-LSTM=0.2201505432、zero=0.2282369311、shuffled=0.2301838175；real 排名第一。random／grouped 的較低分數不進選模 criterion；CV 結果及其 bootstrap 仍是探索性證據。

理由：clustered 稽核比 row-random 更適合減少已知序列相似性重用；real／zero／shuffled 架構參數量相同，有助評估位置對齊結構特徵的開發效益；三個選中 checkpoint 已重播驗證。median 是 L1 誤差所對應的常數 reference，因此主要基準採 median，mean 僅為次要 sanity reference。選定此模型不等於已確認結構特徵的因果效果或宣稱所有模型收斂。

**不在新 test 執行 real-vs-zero／shuffled 確認性比較。** 這些比較保留於開發研究；本協定不授權在看到 test 後加模型、換 seed、換 epoch 或擴充主要假說。

## 2. 鎖定模型與訓練來源

重用 V18 `df2d87bd98ce8ad5b4e27766a98ad73cd5085ada` 已完成的 clustered real checkpoints，seeds=42／123／2026；best epochs=33／35／48，epochs trained=41／43／49。三個檔案的名稱、大小與重新計算的完整 SHA 均在凍結清單。**不重新訓練、不合併 train＋CV、不使用舊 clustered test 擬合、不挑選表現最佳的單一 seed，也不先平均 predictions 做 ensemble。** 這讓當前凍結對象具體且可核查，避免新增最佳化決策。

訓練 lineage 固定：batch=64、Adam lr=0.001、weight decay=1e-5、scheduler patience=2／factor=0.5、最多60 epochs、early-stopping patience=8／min_delta=0.0001、dropout=0.3。CNN channels=64／128、兩層雙向 LSTM hidden=128、兩層 Transformer／8 heads／FFN=1024、輸入7D、輸出2D、max_length=206。以實際 checkpoint 和 archived code 為準；上述設定不是重跑指令。

ViennaRNA=2.7.2，採 `RNA.md()` 預設 model details 與 `RNA.OPTION_MFE`，序列唯一輸入、無量測 constraints。推論特徵必須與已驗證實作及 model details 一致。Kaggle 訓練環境為 Torch 2.10.0+cu128；重播環境須留 Python／Torch／CUDA／ViennaRNA 版本、完整 model details、能量參數來源與檔案 hash（套件內建時記錄套件來源）、feature code commit、每條 sequence 的結構輸出及集合 hash。不得因新 test 結果更換 folding settings。

median／mean 常數只能來自原 V18 clustered train：2A3 median=0.2029999941587448、DMS median=0.10899999737739563；2A3 mean=0.33299498290901597、DMS mean=0.28064647409281696。不得以新 test 或其他分區重新估計。

## 3. 指標與判讀規則

prediction 與 finite target 均 clip 至 [0, 1]；mask 僅含序列長度內、該 profile assay channel 的有限觀測值，缺值與另一 assay channel 不計分。推論使用 `model.eval()`、關閉梯度及既有 padding-safe forward；有效位置出現非有限 prediction 即失敗。

primary metric：每個 checkpoint 對同一個 test 成員與 measured mask 計算 nucleotide-weighted clipped MAE，再取三個 MAE 的算術平均。每個有效量測位置等權；並非兩種 assay 各半，也不是平均 prediction 後的 MAE。

唯一 primary contrast：`median baseline MAE − 三 seed 平均 model MAE`。以新 test similarity cluster 為單位做10,000次配對 percentile bootstrap，RNG seed=17029、雙側95%區間。先在每條 RNA 內合併所有 profiles 的誤差與有效位置，再跨三個 checkpoints 平均誤差分子；每次用同一組 cluster draws 抽模型與 baseline 的完整配對資料。分母亦隨 cluster draws 重算，不能把每個 seed／核苷酸／profile 當獨立抽樣單位。絕對模型 MAE 區間使用同一套 cluster draws。固定使用 `numpy.random.Generator(numpy.random.PCG64(17029))`，按 canonical cluster ID 排序後，每次均勻有放回抽取 K 個 cluster（K 為實際群數）；區間端點使用 `numpy.quantile(..., [0.025, 0.975], method="linear")`，記錄 NumPy 版本及 draws hash。

若 primary contrast 區間下限大於0，才表述「在此預先指定族群與程序下有優於 median baseline 的證據」；否則如實報告未建立優越性。此準則不等於預先保證成功，也沒有設定未經估計的 power。推論條件於三個固定權重及此 cohort 抽樣規則，不代表全新訓練程序的不確定性。

secondary：macro-sequence MAE、各 assay MAE、mean baseline、三個 seed 的個別 MAE 與 sample SD。macro 先合併同一 RNA 的 profiles 再平均；assay 指標只用該 assay 的有效量測位置。secondary 結果描述性呈現，不作額外未校正的確認性成功判定。所有分數與分母均存檔。

## 4. 新 cohort 的固定選取規則

### 4.1 來源與允許的前置資訊

採 Kaggle `stanford-ribonanza-rna-folding/train_data.csv` 的 corrected-current **固定快照**，本地核驗日期為2026-10-22，大小2,364,840,039 bytes、SHA-256：

`527fbcd94ea474054a08346df356ed29381cf2ecb41a84149d8f62c4865a1684`

來源網址與觀察日期記入凍結清單。Kaggle 未提供本次可核驗的上游 dataset version number，因此以快照日期＋完整 SHA 作版本身分，不虛構版號，也不假設今天同名下載仍相同。OLD 僅可用於重建歷史曝露，絕不能供應新 test targets。新下載 SHA 不同即停止並另作協定修訂，不能自動改用新版。

cohort lock 前僅解析 `sequence`、`sequence_id`、`experiment_type`、`SN_filter` 等必要 metadata；全檔 hash 可串流讀 bytes，但不解析 candidate 的 reactivity／reactivity_error，不執行 prediction 或查看 error。SN_filter 是來源提供的品質資格，不宣稱這是完全不依賴量測資訊的抽樣。

### 4.2 完整開發曝露集合 E

E 必須包含所有曾被 train、CV、test、model selection 或 prediction／error 檢視使用的序列，去重後保留逐序列來源與 run provenance。至少包括 V17／V18 的**全部1,000條**，V9／V10 OLD-data cohort，以及所有更早的 smoke／探索 run。不能只用 error CSV 中出現的部分序列推定完整 cohort，也不能只排除 V18 clustered train／CV。

每個歷史 run 記錄：run ID／版本、來源快照 hash、cohort 定義／hash、code commit、曝露類型、可恢復完整序列的證據。相同 cohort 可合併但須保留 lineage。無法完整重建的 run 標記 unresolved；**在 complete-history 核定之前不得產生 eligible final-test manifest**。本次沒有假稱 E 已完整。

### 4.3 確定性的選取

1. 在固定來源中保留 SN_filter==1 的 rows；sequence 必須為非空 ACGU、長度1–206 nt，assay 限2A3_MaP／DMS_MaP。這些規則先固定，不根據預測難度挑選。
2. 先對 sequence 去重。排除 E 中的 exact sequences；再依第5節，排除與 E 任一序列具有 qualifying hit 的 candidate。排除只用 metadata／sequence，不看 reactivity 大小、缺值比例、結構易學程度或 prediction。
3. 對所有剩餘 sequences 計算 `sha256(UTF8("rna-final-2026-11-08-v1\n20261108\n" + sequence))`。其中 `\n` 為一個 LF byte；以該 digest 的小寫十六進位字典序排序，再以 `sha256(UTF8(sequence))` 作 tie-break。取前**1,000條 unique sequences**；原始 CSV row order 不影響排序。
4. 保留入選 RNA 的全部合格 profiles；distinct experimental profiles 的重複 sequence／assay keys 不去重。為每筆 profile 記錄固定來源中的0-based資料 row ordinal（不含 header），讓重複 ID 仍可無歧義重建。
5. 在入選 RNA 內建立第5節指定的 similarity clusters，至少100 clusters；以 metadata 計數，每種 assay 至少有100條不同 RNA。這是最低可分析性檢查，不是經 power analysis 得出的足夠樣本數。固定1,000條是與開發 cohort 同量級、限制測試規模的操作選擇；不得宣稱具有特定 power。
6. 凍結所有成員、profiles、cluster assignments、稽核 receipts 與 hash。若候選不足、cluster／assay floor 不足，**停止，不換 seed、不改 threshold、不多抽直到結果好看**。新方案須於 targets 解鎖前另寫新版本，保留失敗證據。

cohort lock 後才可在獨立 evaluator 解析被選 profiles 的 targets，驗證 schema、每筆至少一個有效量測、無 infinity、無序列外有限 target，保留 NaN 並依既定 mask 處理。如果檢查失敗，整次 evaluation 標為失敗；不可移除該 RNA 再補另一條。來源或已存成員如有變更，所有依賴的 audit 均作廢。

## 5. 相似性稽核與 cluster 定義

新 cohort 使用本次可取得並曾對 V18 frozen split 完成重驗的 **MMseqs2 18-8cc5c+ds-1**，`mmseqs-avx2` 執行檔 SHA 在凍結清單。V17／V18 原始13-45111+ds-2屬歷史 lineage；不能把此次18版結果寫成原13版重建。

固定 search：`easy-search --search-type 3 --strand 1 -k 8 --min-seq-id 0.8 -c 0.8 --cov-mode 0 --alignment-mode 3 --seq-id-mode 0 -s 7.5 --threads 1 --mask 0 --split-memory-limit 4G`，每方向 `--max-seqs=max(300, target序列數)`。輸出 `query,target,fident,qcov,tcov,alnlen,qlen,tlen`；query／target 用 sequence SHA 標記、排序 FASTA、建立新資料庫，不沿用未知 cache。完整實際 argv、執行檔版本／SHA、其他由工具決定的實際參數與 stdout／stderr 均封存。

identity 使用對齊長度作分母，coverage mode0對 query／target 均要求覆蓋門檻；qualifying 以 MMseqs 已執行的固定 filters 為準，不能用輸出小數四捨五入再放寬門檻。所有回傳行必須能解析，且對回已知 FASTA ID；任何傳回 hit 先視為不可忽略的排除／稽核事件，數值或門檻解讀不一致即失敗。forward strand=1是與既有開發稽核一致的操作定義；不保證反向互補或其他生物家族關係分離。

**排除搜尋**：全部 metadata-eligible candidates→完整 E、完整 E→全部 candidates。取兩方向 qualifying hits 涉及的 candidate 聯集加到 exclusion ledger，記錄 candidate／development sequence SHA、方向、identity／coverage／alignment metrics及原因。這一步允許有 hits，因為用途是排除。

**選定後邊界稽核**：新 test→E、E→新 test 必須均為零 qualifying hits，且用選定後最終 FASTA 的 hash。另重新檢查固定 V18 train／CV／新 test 的全部六個有向搜尋。E 包含舊 test 及其他歷史開發資料，所以兩個 E 方向不能被六方向取代。任何方向出錯或有 qualifying hit 都禁止解鎖，不可因另一方向為零而略過。

**test 內 bootstrap clusters**：對選定的1,000條序列做同參數 all-vs-all nucleotide search，把任一方向 qualifying hit 視為無向邊，取 connected components，沒有邊的序列也必須有 singleton cluster；cluster ID 為排序成員 sequence digests 各自加一個 LF 後串接，以 UTF-8 編碼計算 SHA-256。這只定義 bootstrap 的相關性單位，不改變 cohort、重抽樣或訓練模型。記錄命令、原始 hits、cluster成員、群數與大小分布。

可用措辭僅為「在指定 MMseqs2 版本、forward-strand搜尋與門檻下，未偵測到 qualifying cross-boundary hit」，不可宣稱普遍 homology-free。heuristic sensitivity與低於門檻的關係仍是限制。

## 6. Provenance 與 hash 契約

所有 hash 使用 SHA-256；JSON以 UTF-8、sorted keys、indent=2、末尾一個 LF 序列化後對**實際 bytes**計算，CSV／TSV／FASTA 也對實際檔案 bytes 計算。行順序、欄位順序、缺值字面值及 LF 規則必須記錄；不能把不同 cohort-hash 定義直接比較。

| 必備證據 | 來源、版本及必記內容 | 現況 |
|---|---|---|
| 研究決策與模型 | 本協定 ID／日期、freeze commit、3個checkpoint SHA、原訓練commit、V18版本及證據檔SHA | 決策與既有檔案hash已記錄；freeze commit由本次Git紀錄識別 |
| 資料來源 | provider、competition、官方URL、快照觀察日期、實際取得時間、檔名、bytes、SHA | 指定2026-10-22已核驗快照；產生cohort時須再次驗證 |
| 曝露清冊 E | 每個run lineage、完整members、metadata檔及FASTA SHA、complete-history核定紀錄 | 待重建，禁止以V17/V18清冊充當全部歷史 |
| 候選池與排除 | metadata表／排序FASTA SHA、資格rows與unique數、每階段排除數、雙方向結果及ledger SHA | 尚未產生 |
| 入選成員 | 排序規則版本、sequence集合、profile row ordinals、各assay／長度分布、上述檔案SHA | 尚未產生 |
| test clusters | all-vs-all raw hits、canonical members、assignment SHA、cluster群數／分布 | 尚未產生 |
| 邊界audit receipt | 每方向source／target FASTA SHA、工具與binary SHA、argv、exit code、stdout／stderr／TSV SHA、hit數；receipt自身SHA | 尚未產生；零行TSV可代表零hit，但必須有成功exit與完整receipt |
| 測試存取及計分 | 解鎖前所有必要hash已凍結；解鎖後記錄時間、操作者／執行程式、目的、target解析及prediction存取、metrics／bootstrap outputs SHA | 目前未授權；首次寫入的存取紀錄不能冒充未來完整history |

凍結清單中的 `pending_execution_artifacts` 目前為 null，意義是「不存在／未完成」，不是零、空集合或通過。不允許隨機填值、由另一份檔案借用hash，或在缺少原件時只憑說明文字放行。

## 7. 失敗即停止與變更管理

下列任一情況都停止：來源／checkpoint／工具版本或hash不符；歷史曝露不完整；required metadata缺漏或ID歧義；duplicate／missing sequence成員；candidate數不足；cluster或assay最低數量不足；命令exit非零、timeout、OOM、缺檔、未知ID、無法解析或方向不完整；最終邊界出現qualifying hit；targets或有效位置prediction異常；證據hash鏈不完整；cohort在鎖定後改變；或新test已有未記錄的開發曝露。

停止時保留明確原因、時間、執行版本與現有證據，不產生研究指標或合成fallback。參數改變、來源更換、抽樣規則／樣本數／模型／primary contrast改變均需新protocol ID與日期，保留本版，不以覆寫紀錄消除失敗。

**只有規則凍結不代表可跑test。** 下一階段須完成曝露重建、metadata-only cohort builder、fail-closed gate與其回歸驗證，再按既有流程確認test解鎖。當前 `run_ablation.py --run-mode final-eval` 仍無條件拒絕；舊runner內依CV重新選模並訓練的 dormant branch不符合本版「重用三個固定checkpoint」規則，不能直接移除鎖定就執行。

如查看final test後又調整模型或研究決策，該cohort轉列開發曝露，新的確認性結果需另建獨立cohort。任何不利或不確定結果均需保留。
