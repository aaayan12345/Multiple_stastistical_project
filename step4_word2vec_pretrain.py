"""
Step 4: Token 构建 + Word2Vec 预训练 (PyTorch 实现)
4.1 每条交易 → "{行业}_{BUY/SELL}_{金额分档}" token
4.2 构建词表 (token → id)
4.3 Skip-gram + 负采样训练 64 维词向量
"""
import pandas as pd
import numpy as np
import json
import pickle
import sys
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 4.1 加载数据
# ============================================================
print("=" * 60)
print("Step 4: Token 构建 + Word2Vec 预训练 (PyTorch)")
print("=" * 60)

strategies = pd.read_csv('clean_strategies.csv')
accounts = pd.read_csv('clean_accounts.csv')
industry_map = pd.read_csv('stock_industry_mapping.csv')

code2ind = dict(zip(industry_map['stock_code'], industry_map['industry']))
strategies['industry'] = strategies['stock_code'].map(code2ind).fillna('综合')
accounts['industry'] = accounts['stock_code'].map(code2ind).fillna('综合')
strategies['date'] = pd.to_datetime(strategies['datetime'])
accounts['date'] = pd.to_datetime(accounts['datetime'])

# ============================================================
# 4.2 Token 构建
# ============================================================

def build_tokens_for_entity(df):
    """为单个策略/账户构建 token 序列，金额分档按内部三分位数划分"""
    if len(df) == 0:
        return []
    amounts = df['amount'].values
    q33, q66 = np.percentile(amounts, [33.33, 66.67])
    tokens = []
    for _, row in df.sort_values('date').iterrows():
        amt = row['amount']
        if amt <= q33:
            size = 'S'
        elif amt <= q66:
            size = 'M'
        else:
            size = 'L'
        tokens.append(f"{row['industry']}_{row['action']}_{size}")
    return tokens

print("\n--- 构建 Token 序列 ---")

strategy_sequences = {}
for sname in sorted(strategies['strategy_name'].unique()):
    group = strategies[strategies['strategy_name'] == sname]
    strategy_sequences[sname] = build_tokens_for_entity(group)
    print(f"  [{sname}]: {len(strategy_sequences[sname])} tokens")

account_sequences = {}
for aid in sorted(accounts['account_id'].unique()):
    group = accounts[accounts['account_id'] == aid]
    account_sequences[aid] = build_tokens_for_entity(group)
    print(f"  [Account {aid}]: {len(account_sequences[aid])} tokens")

# ============================================================
# 4.3 构建词表
# ============================================================

print("\n--- 构建词表 ---")

all_sequences = list(strategy_sequences.values()) + list(account_sequences.values())
all_tokens = set()
for seq in all_sequences:
    all_tokens.update(seq)

sorted_tokens = sorted(all_tokens)
token2id = {tok: i for i, tok in enumerate(sorted_tokens)}
id2token = {i: tok for tok, i in token2id.items()}
VOCAB_SIZE = len(token2id)

print(f"  词表大小: {VOCAB_SIZE}")

# 统计
ind_token_count = defaultdict(int)
for seq in all_sequences:
    for tok in seq:
        ind = tok.rsplit('_', 2)[0]
        ind_token_count[ind] += 1

print("\n  Token 分布 (按行业):")
for ind in sorted(ind_token_count.keys()):
    print(f"    {ind:8s}: {ind_token_count[ind]:4d}")

# ============================================================
# 4.4 Skip-gram Word2Vec (PyTorch)
# ============================================================

print("\n--- Skip-gram Word2Vec 训练 (PyTorch) ---")

VECTOR_SIZE = 64
WINDOW = 5
NEG_SAMPLES = 5
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.002

# ----- 构建训练数据：Skip-gram (center, context) 正样本对 -----
print("  构建 Skip-gram 正样本对...")
pos_pairs = []  # list of (center, context)
for seq in all_sequences:
    ids = [token2id[t] for t in seq]
    for i, center in enumerate(ids):
        start = max(0, i - WINDOW)
        end = min(len(ids), i + WINDOW + 1)
        for j in range(start, end):
            if i != j:
                pos_pairs.append((center, ids[j]))

pos_pairs = np.array(pos_pairs, dtype=np.int64)
num_pairs = len(pos_pairs)
print(f"  正样本对数: {num_pairs}")

# 负采样分布: 基于词频^0.75 (标准做法)
token_counts = np.zeros(VOCAB_SIZE, dtype=np.float32)
for seq in all_sequences:
    for tok in seq:
        token_counts[token2id[tok]] += 1
noise_dist = token_counts ** 0.75
noise_dist /= noise_dist.sum()


# ----- 模型定义 -----
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)   # 中心词向量 v_c
        self.out_embed = nn.Embedding(vocab_size, embed_dim)  # 上下文词向量 u_o

    def forward(self, center, context, neg_samples):
        """返回正样本和负样本的 logits"""
        v_c = self.in_embed(center)        # (batch, dim)
        u_pos = self.out_embed(context)    # (batch, dim)
        u_neg = self.out_embed(neg_samples)  # (batch, neg, dim)

        pos_score = (v_c * u_pos).sum(dim=1)  # (batch,)
        neg_score = (v_c.unsqueeze(1) * u_neg).sum(dim=2)  # (batch, neg)

        return pos_score, neg_score


device = torch.device('cpu')
model = SkipGram(VOCAB_SIZE, VECTOR_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"  vector_size={VECTOR_SIZE}, window={WINDOW}, neg_samples={NEG_SAMPLES}, "
      f"epochs={EPOCHS}, batch={BATCH_SIZE}")

# ----- 训练循环 -----
model.train()
total_batches = num_pairs // BATCH_SIZE

for epoch in range(1, EPOCHS + 1):
    # 每个 epoch 打乱正样本对
    perm = torch.randperm(num_pairs)
    epoch_loss = 0.0

    for b in range(total_batches):
        idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        center_b = torch.tensor(pos_pairs[idx, 0], dtype=torch.long, device=device)
        context_b = torch.tensor(pos_pairs[idx, 1], dtype=torch.long, device=device)

        # 负采样 (从 noise_dist 中采样)
        neg_b = torch.multinomial(
            torch.tensor(noise_dist, dtype=torch.float32),
            BATCH_SIZE * NEG_SAMPLES,
            replacement=True
        ).view(BATCH_SIZE, NEG_SAMPLES).to(device)

        pos_score, neg_score = model(center_b, context_b, neg_b)

        # Binary cross-entropy loss
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).mean()
        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        avg_loss = epoch_loss / max(total_batches, 1)
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}")

# 取中心词向量作为最终 embedding (常见做法)
embedding_matrix = model.in_embed.weight.detach().cpu().numpy()
print(f"  Embedding 矩阵: {embedding_matrix.shape}")

# ============================================================
# 4.5 保存
# ============================================================

print("\n--- 保存 ---")

with open('token_vocab.json', 'w', encoding='utf-8') as f:
    json.dump({'token2id': token2id, 'id2token': {str(k): v for k, v in id2token.items()}},
              f, ensure_ascii=False, indent=2)

np.save('word2vec_embeddings.npy', embedding_matrix)

tokenized = {
    'strategies': {s: [token2id[t] for t in seq] for s, seq in strategy_sequences.items()},
    'accounts': {a: [token2id[t] for t in seq] for a, seq in account_sequences.items()},
}
with open('tokenized_sequences.pkl', 'wb') as f:
    pickle.dump(tokenized, f)

seq_rows = []
for entity_type, entities in [('strategy', strategy_sequences), ('account', account_sequences)]:
    for name, tokens in entities.items():
        seq_rows.append({
            'entity_type': entity_type,
            'entity_name': name,
            'seq_length': len(tokens),
            'token_ids': ','.join(str(token2id[t]) for t in tokens),
        })
pd.DataFrame(seq_rows).to_csv('token_sequences.csv', index=False, encoding='utf-8-sig')

torch.save(model.state_dict(), 'word2vec_model.pt')

print(f"  token_vocab.json         — 词表 ({VOCAB_SIZE} tokens)")
print(f"  word2vec_embeddings.npy  — {embedding_matrix.shape} embedding 矩阵")
print(f"  tokenized_sequences.pkl   — token id 序列 (Step 6 用)")
print(f"  token_sequences.csv      — 可读版序列")
print(f"  word2vec_model.pt        — PyTorch 模型权重")

# ============================================================
# 4.6 语义验证：查询最相似 tokens
# ============================================================

print("\n--- 语义验证: 查询最相似 tokens ---")

# 归一化以计算 cosine 相似度
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
emb_norm = embedding_matrix / (norms + 1e-10)

def most_similar(token, topn=5):
    if token not in token2id:
        return []
    tid = token2id[token]
    vec = emb_norm[tid]
    sims = emb_norm @ vec  # cosine similarity (归一化后 dot product = cosine)
    sims[tid] = -1  # 排除自己
    top_ids = np.argsort(-sims)[:topn]
    return [(id2token[i], float(sims[i])) for i in top_ids]

test_queries = ['电子_BUY_L', '银行_BUY_S', '食品饮料_SELL_M', '国防军工_BUY_M']
for query in test_queries:
    print(f"\n  与 '{query}' 最相似:")
    similar = most_similar(query, topn=5)
    for tok, score in similar:
        print(f"    {tok:32s}  sim={score:.4f}")

print("\nDone. Step 4 完成.")
