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
│
├── 绩效1.xlsx                         # 原始数据：12 个策略交易记录
├── 绩效2.xlsx                         # 原始数据：22 个策略交易记录
├── 模拟账户A.xlsx                     # 原始数据：模拟账户 A 交易记录
├── 模拟账户B.xlsx                     # 原始数据：模拟账户 B 交易记录
├── 模拟账户C.xlsx                     # 原始数据：模拟账户 C 交易记录
│
├── step1_data_loader.py               # Step 1: 数据加载与清洗
├── step2_industry_mapping.py          # Step 2: 股票→行业映射
├── step3_feature_extraction.py        # Step 3: 6 维特征提取
│
├── clean_strategies.csv               # [产出] 清洗后策略交易记录
├── clean_accounts.csv                 # [产出] 清洗后账户交易记录
├── stock_industry_mapping.csv         # [产出] 股票代码→申万行业映射表
├── strategy_features.csv              # [产出] 34 策略特征向量
├── strategy_features.json             # [产出] 34 策略特征向量 (JSON)
├── account_features.csv               # [产出] 3 账户特征向量
├── account_features.json              # [产出] 3 账户特征向量 (JSON)
│
├── _step1_result.txt                  # Step 1 运行摘要
├── _step2_final.txt                   # Step 2 运行摘要（规则 + DeepSeek API）
├── _step2_test_noapi.py               # Step 2 纯规则测试版（不需 API）
│
├── step4_word2vec_pretrain.py         # [TODO] Step 4: Token构建 + Word2Vec
├── step5_simulate_data.py             # [TODO] Step 5: 模拟数据生成
├── step6_lstm_contrastive_train.py    # [TODO] Step 6: LSTM + 对比学习
├── step7_match_evaluate.py            # [TODO] Step 7: 匹配评估 + 归因分析
│
└── models/                            # [TODO] 保存的模型文件
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

**状态**：📋 待完成

**做什么**：
- 将每条交易记录编码为 token：`{行业}_{买卖方向}_{金额分档}`
- 金额分 3 档（S/M/L），按策略内部金额分布的三分位数划分
- 词表总量约 31×2×3 = 186 个 token（实际约 150+）
- 用 Skip-gram 训练 64 维词向量，使语义相近的 token 在向量空间中靠近
- 窗口大小 5，负采样，epochs=10

**产出文件**：
| 文件 | 内容 |
|------|------|
| `token_vocab.json` | token→id 映射表 |
| `word2vec_embeddings.npy` | 64 维词向量矩阵 |
| `tokenized_sequences.pkl` | 各策略/账户的 token 序列 |

**脚本**：`step4_word2vec_pretrain.py`

---

### Step 5 — 模拟数据生成

**状态**：📋 待完成

**做什么**：
- 基于 Step 3 提取的 34 策略特征分布，用高斯扰动生成 500 个模拟策略
- 基于 3 个账户特征分布，生成 200 个模拟客户
- 建立"模拟策略 ↔ 模拟客户"的已知匹配关系作为训练标签
- 用采样 + 噪声注入方式生成每对匹配实体的交易序列
- 模拟策略和模拟客户作为训练集，3 个真实客户作为测试集

**产出文件**：
| 文件 | 内容 |
|------|------|
| `simulated_strategies.csv` | 500 模拟策略特征 + token序列 |
| `simulated_accounts.csv` | 200 模拟客户特征 + token序列 |
| `train_pairs.csv` | 训练用匹配/不匹配对标签 |

**脚本**：`step5_simulate_data.py`

---

### Step 6 — LSTM 编码器 + 对比学习训练

**状态**：📋 待完成

**做什么**：
- LSTM Encoder 把不定长 token 序列编码为固定 128 维向量
- 对比学习：Triplet Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
- Anchor: 模拟策略向量，Positive: 匹配的模拟客户向量，Negative: 随机不匹配客户
- 训练后，任意策略和账户的编码向量可通过 Cosine 相似度计算匹配分数
- 验证集上监控 matching accuracy

**产出文件**：
| 文件 | 内容 |
|------|------|
| `models/lstm_encoder.pt` | 训练好的 LSTM 编码器权重 |
| `models/training_log.csv` | 训练 loss/accuracy 日志 |
| `strategy_embeddings.npy` | 34 策略的 128 维编码向量 |
| `account_embeddings.npy` | 3 账户的 128 维编码向量 |

**脚本**：`step6_lstm_contrastive_train.py`

---

### Step 7 — 匹配评估 + 归因分析

**状态**：📋 待完成

**做什么**：
- 计算 3 个真实客户与 34 个策略的 Cosine 相似度矩阵
- 每个客户输出 Top-3 推荐策略及匹配置信度
- 与队友 Phase 1 传统统计方法结果做 Spearman 交叉验证
- SHAP 归因分析：6 个特征中哪个对匹配贡献最大
- 输出匹配报告

**产出文件**：
| 文件 | 内容 |
|------|------|
| `match_results.csv` | 3 客户 × 34 策略相似度矩阵 |
| `match_report.txt` | Top-3 推荐 + 置信度 |
| `attribution.png` | SHAP 特征重要性图 |
| `cross_validation.csv` | 与传统方法对比 |

**脚本**：`step7_match_evaluate.py`

---

## 技术栈

- **数据**：pandas, numpy, akshare
- **API**：DeepSeek API (OpenAI SDK, model: deepseek-chat)
- **深度学习**：PyTorch, gensim (Word2Vec)
- **评估**：scikit-learn (cosine_similarity), SHAP
- **语言**：Python 3.x

---

## 运行方式

```bash
# Step 1-3: 数据准备（已完成）
python step1_data_loader.py
python step2_industry_mapping.py    # 需要 DeepSeek API key
python step3_feature_extraction.py

# Step 4-7: 模型训练与评估（待完成）
python step4_word2vec_pretrain.py
python step5_simulate_data.py
python step6_lstm_contrastive_train.py
python step7_match_evaluate.py
```
