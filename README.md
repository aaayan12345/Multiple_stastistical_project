# 证券投资策略与客户交易画像精准匹配研究

## Phase 2: 深度学习辅助赛道

基于 Word2Vec Token Embedding + LSTM Encoder + 对比学习（Contrastive Learning），
将 3 个模拟客户账户与 34 个量化策略进行精准匹配。

---

## 项目结构

```
project/
│
├── README.md                          # 本文件
├── .env.example                       # DeepSeek API Key 模板
│
├── 绩效1.xlsx                         # 原始数据：12 个策略交易记录
├── 绩效2.xlsx                         # 原始数据：22 个策略交易记录
├── 模拟账户A.xlsx                     # 原始数据：模拟账户 A 交易记录
├── 模拟账户B.xlsx                     # 原始数据：模拟账户 B 交易记录
├── 模拟账户C.xlsx                     # 原始数据：模拟账户 C 交易记录
│
│── step1_data_loader.py               # Step 1: 数据加载与清洗
│── step2_industry_mapping.py          # Step 2: 股票→行业映射
│── step3_feature_extraction.py        # Step 3: 6 维特征提取
│── step4_word2vec_pretrain.py         # Step 4: Token构建 + Word2Vec
│── step5_simulate_data.py             # Step 5: 模拟数据生成
│── step6_lstm_contrastive.py          # Step 6: LSTM编码器 + 对比学习
│── step7_evaluation.py                # Step 7: 匹配评估 + 归因分析
│
│── clean_strategies.csv               # [产出 Step1] 清洗后策略交易记录
│── clean_accounts.csv                 # [产出 Step1] 清洗后账户交易记录
│── stock_industry_mapping.csv         # [产出 Step2] 股票代码→申万行业映射
│── strategy_features.csv/json         # [产出 Step3] 34 策略特征向量
│── account_features.csv/json          # [产出 Step3] 3 账户特征向量
│
│── token_vocab.json                   # [产出 Step4] Token→ID 映射 (221 tokens)
│── word2vec_embeddings.npy            # [产出 Step4] 221×64 词向量矩阵
│── tokenized_sequences.pkl            # [产出 Step4] Token ID 序列
│── token_sequences.csv                # [产出 Step4] 可读版序列
│── word2vec_model.pt                  # [产出 Step4] PyTorch 模型权重
│
│── simulated_strategies_features.csv  # [产出 Step5] 500 模拟策略特征
│── simulated_accounts_features.csv    # [产出 Step5] 200 模拟客户特征
│── simulated_data.pkl                 # [产出 Step5] 完整模拟数据
│── train_pairs.csv                    # [产出 Step5] 1000 条训练标签
│── simulated_sequences.csv            # [产出 Step5] 模拟序列
│
│── models/lstm_encoder.pt             # [产出 Step6] 训练好的 LSTM 编码器
│── strategy_embeddings.npy            # [产出 Step6] 34×128 策略向量
│── account_embeddings.npy             # [产出 Step6] 3×128 账户向量
│── embedding_meta.json                # [产出 Step6] 向量名映射
│── training_history.csv               # [产出 Step6] 训练 loss/acc 日志
│
│── similarity_matrix.csv              # [产出 Step6] 3×34 匹配相似度矩阵
│── matching_phase1_features.csv       # [产出 Step7] Phase 1 特征匹配矩阵
│── matching_phase2_lstm.csv           # [产出 Step7] Phase 2 LSTM 匹配矩阵
│── final_recommendations.csv          # [产出 Step7] 综合推荐 Top-5
│── shap_analysis.json                 # [产出 Step7] SHAP 特征归因结果
│
│── _step1_result.txt                  # Step 1 运行摘要
│── _step2_final.txt                   # Step 2 运行摘要
│── _step2_test_noapi.py               # Step 2 纯规则测试版
│
└── models/                            # 模型保存目录
```

---

## 七步流程

### Step 1 — 数据加载与清洗

**做什么**：合并绩效1（12 个策略）和绩效2（22 个策略），共 34 个策略的 31,686 条交易记录；
加载 3 个模拟账户 A/B/C 共 2,130 条交易记录。统一列名、处理缺失值、过滤无效记录。

**关键处理**：
- 列名映射统一（`datetime`, `stock_code`, `action`, `volume`, `price`, `amount`）
- 排除无交易记录的 6 个策略（策略ETF, 策略etf2, 全球etf增强, 百亿etf等）
- 过滤非主动买卖事件（配售股份、中签下账、新股入账）
- 股票代码标准化（去前缀 SHSE/SZSE，补零至 6 位）
- 金额列类型转换（混合字符串/浮点数 → 统一 float）
- 排除可转债/债券/B股（代码 12xxxx, 11xxxx, 10xxxx, 9xxxxx 等）

**产出文件**：
| 文件 | 内容 | 行数 |
|------|------|------|
| `clean_strategies.csv` | 清洗后策略交易记录 | 31,686 |
| `clean_accounts.csv` | 清洗后账户交易记录 | 2,130 |

**脚本**：`step1_data_loader.py`

---

### Step 2 — 股票代码 → 申万一级行业映射

**做什么**：将 2,708 只去重股票映射到 31 个申万一级行业，为 Step 3 行业偏好特征提供基础。

**四轮分类策略**：

| 轮次 | 方法 | 匹配数 | 说明 |
|------|------|--------|------|
| 第一轮 | 关键词规则 | 604 | 正则匹配股票名称（如"煤业"→煤炭、"半导体"→电子） |
| 第二轮 | 策略名推断 | 312 | 从策略名推断行业（如"军工etf增强"→国防军工） |
| 第三轮 | ETF代码段 | 0 | ETF代码前缀匹配（15xxxx/51xxxx 等→综合） |
| 第四轮 | DeepSeek API | 1,792 | 批量查询未匹配 + 综合类股票（每批 30 只，100%成功率） |

**最终覆盖**：2,708 只股票 100% 映射，0 只遗漏。

**行业分布 Top 5**：电子(377)、医药生物(234)、基础化工(173)、机械设备(171)、计算机(167)

**产出文件**：
| 文件 | 内容 | 列 |
|------|------|-----|
| `stock_industry_mapping.csv` | 股票→行业映射 | `stock_code`, `stock_name`, `industry`, `source` |

**脚本**：`step2_industry_mapping.py`（需 DeepSeek API）、`_step2_test_noapi.py`（纯规则测试版）

---

### Step 3 — 6 维交易风格特征提取

**做什么**：把每个策略和账户的交易行为抽象为 6 个可量化的数值特征，构建固定维度的"交易画像向量"，作为后续深度学习的先验。

**6 个特征**：

| 特征 | 维度 | 捕捉什么 | 计算方式 |
|------|------|---------|---------|
| F1 行业偏好 | 31 | 钱投向哪些行业 | 买入金额按行业分布 + Laplace 平滑 |
| F2 年化换手率 | 1 | 交易频率 | 总买入 / 平均持仓市值 / 时间跨度(年) |
| F3 持仓周期 | 1 | 买入到卖出多久 | FIFO 匹配 → 成交量加权平均持有天数 |
| F4 持股集中度 | 1 | 分散还是集中 | 时序仓位 HHI 指数均值 |
| F5 买卖对称性 | 1 | 建仓还是清仓 | 总买入 / (总买入+总卖出)，0.5=平衡 |
| F6 波动偏好 | 1 | 喜欢高波动还是低波动 | 各股价格变异系数 CoV 加权平均 |

**最终向量**：每个策略/账户 → **36 维向量**（31 行业概率 + 5 数值特征）

**三个账户画像**：

| 特征 | 账户 A | 账户 B | 账户 C |
|------|--------|--------|--------|
| 换手率 | 5.9 | 89.2 | 36.9 |
| 持仓周期 | 56.8 天 | 3.1 天 | 7.3 天 |
| 集中度 HHI | 0.47 | 0.29 | 0.38 |
| 买卖对称 | 0.54（净买） | 0.50（平衡） | 0.50（平衡） |
| 波动偏好 | 0.07 | 0.13 | 0.04 |

**产出文件**：
| 文件 | 内容 |
|------|------|
| `strategy_features.csv` | 34 策略特征 |
| `strategy_features.json` | 同上 JSON 格式 |
| `account_features.csv` | 3 账户特征 |
| `account_features.json` | 同上 JSON 格式 |

**脚本**：`step3_feature_extraction.py`

---

### Step 4 — Token 构建 + Word2Vec 预训练

**状态**：✅ 已完成

**做什么**：
- 每条交易记录 → `{行业}_{BUY/SELL}_{S/M/L}` token，金额分档按策略内部三分位数
- 词表共 **221 个 token**，覆盖 31 个申万行业 × 2 方向 × 3 档金额
- PyTorch 从零实现 Skip-gram + 负采样 Word2Vec（非 gensim，避免 Windows C++ 编译器依赖）
- 64 维词向量，窗口=5，负采样=5，30 epochs，Adam lr=0.002
- 训练集 337,050 对正样本，loss 从 3.43 → 1.08

**语义验证**（余弦相似度查询）：

| 查询 Token | Top-2 相似 Token |
|-----------|-----------------|
| `电子_BUY_L` | 电子_SELL_L (0.92), 计算机_BUY_L (0.78) |
| `银行_BUY_S` | 银行_SELL_S (0.89), 非银金融_BUY_S (0.75) |
| `食品饮料_SELL_M` | 食品饮料_BUY_M (0.85), 商贸零售_SELL_M (0.72) |

**产出文件**：
| 文件 | 内容 |
|------|------|
| `token_vocab.json` | Token→ID 映射 (221 tokens) |
| `word2vec_embeddings.npy` | 221×64 词向量矩阵 |
| `tokenized_sequences.pkl` | 各策略/账户的 token ID 序列 |
| `token_sequences.csv` | 可读版序列 |
| `word2vec_model.pt` | PyTorch 模型权重 |

**脚本**：`step4_word2vec_pretrain.py`

---

### Step 5 — 模拟数据生成

**状态**：✅ 已完成

**做什么**：
- 基于 34 策略 + 3 账户的真实特征分布，扰动生成模拟数据用于对比学习训练
- **特征扰动**：数值特征用对数正态噪声（σ=0.12~0.15），行业偏好用 Dirichlet 噪声
- **序列生成**：Block bootstrap 从真实序列采样 token 块（3~12 tokens），按行业偏好加权拼接，并微调 BUY/SELL 比例
- **匹配标签**：余弦相似度 top-1 作为正样本，bottom-50% 随机 4 个作为负样本（200 正 + 800 负 = 1000 对）

**生成规模**：

| 实体 | 数量 | 序列长度 (mean/min/max) |
|------|------|------------------------|
| 模拟策略 | 500 | 1,344 / 542 / 2,867 |
| 模拟客户 | 200 | 1,105 / 407 / 1,610 |
| 正样本平均相似度 | — | 0.894 |

**产出文件**：
| 文件 | 内容 |
|------|------|
| `simulated_strategies_features.csv` | 500 模拟策略 36 维特征 |
| `simulated_accounts_features.csv` | 200 模拟客户 36 维特征 |
| `simulated_data.pkl` | 完整模拟数据 (特征 + 序列 + 匹配标签) |
| `train_pairs.csv` | 1000 条训练标签 (client_idx, strategy_idx, is_match) |
| `simulated_sequences.csv` | 模拟 token 序列 |

**脚本**：`step5_simulate_data.py`

---

### Step 6 — LSTM 编码器 + 对比学习训练

**状态**：✅ 已完成

**做什么**：
- 构建序列编码器：`Word2Vec Embedding(221×64) → BiLSTM(2层, hidden=128) → Mean Pooling → Linear(256→128) → L2 归一化`
- 总参数 657,536，Word2Vec 预训练权重初始化 Embedding 层
- 对比学习 Triplet Loss：`max(0, d(anchor, pos) - d(anchor, neg) + 0.5)`
- 训练时随机截取 300-token 子序列（数据增强）
- 50 epochs, batch=32, Adam lr=0.001, CosineAnnealing 调度
- 训练集 160 客户，验证集 40 客户

**训练结果**：最佳 val_loss=0.301, val_acc 最高 92.5% (Epoch 40)

**产出文件**：
| 文件 | 内容 |
|------|------|
| `models/lstm_encoder.pt` | 训练好的编码器 (含配置+训练历史) |
| `strategy_embeddings.npy` | 34×128 真实策略向量 |
| `account_embeddings.npy` | 3×128 真实账户向量 |
| `similarity_matrix.csv` | 3×34 余弦相似度矩阵 |
| `training_history.csv` | 50 轮训练 loss/acc |

**脚本**：`step6_lstm_contrastive.py`

---

### Step 7 — 匹配评估 + 归因分析

**状态**：✅ 已完成

**做什么**：
- **Phase 1 Baseline**（队友并行进行）：36 维特征余弦相似度匹配，作为对照基线
- **Phase 2 深度学习**：LSTM 128 维向量余弦相似度匹配
- **两阶段对比**：Spearman 秩相关、Top-K 重叠率、排名变化分析
- **SHAP 归因**：在 1000 对模拟数据上训练 XGBoost，用 SHAP TreeExplainer 解释特征贡献

**关键发现**：

| 指标 | Phase 1 (特征工程) | Phase 2 (LSTM) |
|------|-------------------|----------------|
| 相似度范围 | -0.30 ~ 0.40 | -0.81 ~ 1.00 |
| 相似度标准差 | 0.17 | 0.60 |
| Spearman ρ (A/B/C) | — | 0.72 / 0.40 / 0.44 |

**SHAP Top-5 驱动特征**：持仓周期 > 集中度 > 买卖对称性 > 换手率 > 波动偏好（行业偏好贡献极低）

**综合推荐** (Phase 1 + Phase 2 平均排名)：

| 排名 | Account A | Account B | Account C |
|------|-----------|-----------|-----------|
| 1 | 煤炭周期优选动态轮动 | 行业etf增强 | 中证1000增强 |
| 2 | 成长红利量化选股 | 动量趋势策略 | etf动量改 |
| 3 | 杠铃 | 综合全 | 综合全 |
| 4 | 化工ETF优选 | 综合拆分1 | 行业etf增强 |
| 5 | 食品etf增强 | etf动量改 | 综合拆分2 |

**产出文件**：
| 文件 | 内容 |
|------|------|
| `matching_phase1_features.csv` | Phase 1 特征匹配矩阵 |
| `matching_phase2_lstm.csv` | Phase 2 LSTM 匹配矩阵 |
| `final_recommendations.csv` | 综合推荐 Top-5 × 3 账户 |
| `shap_analysis.json` | SHAP 特征归因结果 |

**脚本**：`step7_evaluation.py`

---

## 技术栈

- **数据**：pandas, numpy
- **API**：DeepSeek API (OpenAI SDK, model: deepseek-v4pro)
- **深度学习**：PyTorch (Word2Vec Skip-gram, BiLSTM Encoder)
- **评估**：scikit-learn, SciPy, SHAP, XGBoost
- **语言**：Python 3.x
- **GPU 支持**：代码自动检测 `cuda` 设备，可部署至 GPU 服务器重新训练

---

## 运行方式

```bash
# Step 1-3: 数据准备
python step1_data_loader.py
python step2_industry_mapping.py    # 需要 DeepSeek API key
python step3_feature_extraction.py

# Step 4-7: 模型训练与评估
python step4_word2vec_pretrain.py
python step5_simulate_data.py
python step6_lstm_contrastive.py
python step7_evaluation.py
```

## Phase 说明

| Phase | 方法 | 负责人 | 状态 |
|-------|------|--------|------|
| Phase 1 | 传统统计匹配（特征工程 + 聚类 + 多度量） | 队友 | 进行中 |
| Phase 2 | 深度学习匹配（Word2Vec + BiLSTM + 对比学习） | 本项目 | ✅ 已完成 |
