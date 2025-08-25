import numpy as np
import torch

# 生成输入矩阵X (batch_size, sentence_len, model_len)
X = torch.randn(size=(32, 32, 64))
mask = torch.randint(low=0, high=5, size=(32,1))
for i in range(len(mask)): 
    mask_value = mask[i]
    matrix = X[i]
    for col in range(mask_value):
        idx = -(col + 1)
        matrix[idx, :] = 0
    X[i] = matrix


# X = (b, s, m)
# Q = X @ Wq 
# K = X @ Wk 
# V = X @ Wv
# Wq = (model_len, Kq)
# Wk = (model_len, Kq)
# Q = (b, s, m) @ (m, Kq) = (b, s, Kq) 
# K = (b, s, m) @ (m, Kq) = (b, s, Kq)
# T = Q @ K.transpose(1, 2) = (b, s, s)

# # 权重矩阵维度
# Wq = (m, m)  # 实际实现中通常为 (m, m) 而不是 (m, d_k)
# Wk = (m, m)
# Wv = (m, m)

# # 线性变换
# Q = X @ Wq  # (b, s, m) @ (m, m) -> (b, s, m)
# K = X @ Wk  # (b, s, m) @ (m, m) -> (b, s, m)
# V = X @ Wv  # (b, s, m) @ (m, m) -> (b, s, m)

# 重塑和转置Q、K、V
# Q = Q.reshape(b, s, h, d_k).transpose(1, 2)  # (b, h, s, d_k)
# K = K.reshape(b, s, h, d_k).transpose(1, 2)  # (b, h, s, d_k)
# V = V.reshape(b, s, h, d_k).transpose(1, 2)  # (b, h, s, d_k)



# attn_scores = Q @ K.transpose(-2, -1)  # (b, h, s, d_k) @ (b, h, d_k, s) -> (b, h, s, s)
# attn_scores = attn_scores / math.sqrt(d_k)
# attn_weights = F.softmax(attn_scores, dim=-1)  # (b, h, s, s)
# attn_output = attn_weights @ V  # (b, h, s, s) @ (b, h, s, d_k) -> (b, h, s, d_k)