import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# 单头注意力
seq_len = 4
d_k = 8
d_v = 8

Q = torch.randn(4, 8)
K = torch.randn(4, 8)
V = torch.randn(4, 8)

# S.shape=(4, 4),A.shape=(4, 4)
S = Q @ K.T / d_k**0.5
A = torch.softmax(S, dim=1)

# O.shape=(4, 8)
O = A @ V

print("Attention Scores:\n", S)
print("\nAttention Weights:\n", A)

# 热力图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im1 = axes[0].imshow(S, cmap="viridis")
axes[0].set_title("Attention Scores")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(A, cmap="viridis", vmin=0, vmax=1)
axes[1].set_title("Attention Weights")
fig.colorbar(im2, ax=axes[1])

for ax in axes:
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))

plt.show()

# 多头注意力
seq_len = 4
d_model = 8
num_heads = 2
d_k = d_model // num_heads
d_v = d_model // num_heads

X = torch.randn(seq_len, d_model)
Wq1 = torch.randn(d_k, d_model) / d_model**0.5
Wk1 = torch.randn(d_k, d_model) / d_model**0.5
Wv1 = torch.randn(d_v, d_model) / d_model**0.5
Wq2 = torch.randn(d_k, d_model) / d_model**0.5
Wk2 = torch.randn(d_k, d_model) / d_model**0.5
Wv2 = torch.randn(d_v, d_model) / d_model**0.5

Q1 = X @ Wq1.T
K1 = X @ Wk1.T
V1 = X @ Wv1.T
Q2 = X @ Wq2.T
K2 = X @ Wk2.T
V2 = X @ Wv2.T

S1 = Q1 @ K1.T / d_k**0.5
A1 = torch.softmax(S1, dim=1)
O1 = A1 @ V1
S2 = Q2 @ K2.T / d_k**0.5
A2 = torch.softmax(S2, dim=1)
O2 = A2 @ V2

O_concat = torch.cat([O1, O2], dim=1)
Wo = torch.randn(d_model, d_model) / d_model**0.5
O = O_concat @ Wo.T

print("\nHead 1 Scores:\n", S1)
print("\nHead 1 Weights:\n", A1)
print("\nHead 2 Scores:\n", S2)
print("\nHead 2 Weights:\n", A2)

# 多头注意力热力图
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

scores = [S1, S2]
weights = [A1, A2]

for i in range(num_heads):

    im1 = axes[i, 0].imshow(
        scores[i],
        cmap="viridis"
    )

    axes[i, 0].set_title(f"Head {i + 1} - Scores")
    fig.colorbar(im1, ax=axes[i, 0])

    im2 = axes[i, 1].imshow(
        weights[i],
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[i, 1].set_title(f"Head {i + 1} - Weights")
    fig.colorbar(im2, ax=axes[i, 1])

for ax in axes.flat:
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))

plt.show()