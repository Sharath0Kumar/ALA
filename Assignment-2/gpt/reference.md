# GPT-2 Small — Formal Algorithms Reference

Source: *Formal Algorithms for Transformers*, Phuong and Hutter (DeepMind, 2022), [arXiv:2207.09238](https://arxiv.org/abs/2207.09238).

## GPT-2 Small hyperparameters

| Symbol | Value | Meaning |
|---|---:|---|
| `l_max` | 1024 | Maximum sequence length |
| `d_e` | 768 | Embedding / model width |
| `L` | 12 | Number of transformer layers |
| `H` | 12 | Number of attention heads |
| `N_V` | 50,257 | Vocabulary size |
| `d_mlp` | 3072 | MLP hidden width (`4 * d_e`) |
| `d_attn` | 64 | Per-head attention dimension (`d_e / H`) |
| `d_mid` | 64 | Per-head value dimension (alias of `d_attn`) |

Derived relations:

```text
d_attn = d_mid = d_e / H = 768 / 12 = 64
H * d_mid = 12 * 64 = 768 = d_e
```

## Algorithm 1 — Token embedding

**Input:** Token ID `v` (integer in `[0, N_V)`)  
**Output:** Token vector `e` in `R^d_e`  
**Parameter:** `W_e` in `R^(d_e x N_V)`

```text
e = W_e[:, v]
```

`W_e` stores one `d_e`-dimensional column per vocabulary token. Selecting column `v` gives that token's embedding.

GPT-2 Small shapes:

```text
W_e : (768, 50257)
v   : integer in [0, 50256]
e   : (768,)
```

## Algorithm 6 — Layer normalization

**Input:** Activations `e` in `R^d_e`  
**Output:** Normalized activations `e_hat` in `R^d_e`  
**Parameters:** Learned scale `gamma` and offset `beta`, each in `R^d_e`

```text
m     = (1 / d_e) * SUM(i = 1..d_e) e[i]
v     = (1 / d_e) * SUM(i = 1..d_e) (e[i] - m)^2
e_hat = ((e - m) / sqrt(v + eps)) * gamma + beta
```

The multiplication by `gamma` is element-wise. Both `m` and `v` are scalars computed over all `d_e` elements. Use `eps = 1e-5` for numerical stability.

NumPy mapping:

```python
m = np.sum(e) / e.shape[0]
v = np.sum((e - m) ** 2) / e.shape[0]
e_hat = ((e - m) / np.sqrt(v + eps)) * gamma + beta
```

## Algorithm 4 — Attention (single head)

**Input:**

- `X` in `R^(d_x x l_x)`: primary sequence; queries come from `X`
- `Z` in `R^(d_z x l_z)`: context sequence; keys and values come from `Z`

**Output:** `V_tilde` in `R^(d_out x l_x)`, updated token representations

Parameters:

| Parameter | Shape | Purpose |
|---|---|---|
| `W_q`, `b_q` | `R^(d_attn x d_x)`, `R^d_attn` | Query projection |
| `W_k`, `b_k` | `R^(d_attn x d_z)`, `R^d_attn` | Key projection |
| `W_v`, `b_v` | `R^(d_out x d_z)`, `R^d_out` | Value projection |

The mask has shape `(l_z, l_x)`. `Mask[t_z, t_x] = 1` allows query position `t_x` to attend to context position `t_z`; `0` masks that score to negative infinity before softmax.

For GPT-2's causal self-attention:

```text
Mask[t_z, t_x] = 1 iff t_z <= t_x
```

Pseudocode:

```text
Q = W_q * X + b_q * 1^T_lx       # (d_attn, l_x)
K = W_k * Z + b_k * 1^T_lz       # (d_attn, l_z)
V = W_v * Z + b_v * 1^T_lz       # (d_out, l_z)
S = K^T * Q / sqrt(d_attn)        # (l_z, l_x)

S[t_z, t_x] = -inf where Mask[t_z, t_x] = 0
V_tilde = V * softmax_colwise(S)  # (d_out, l_x)
```

`1^T` is a row vector of ones, which broadcasts each bias across sequence positions. Each score is `k[t_z]^T * q[t_x]`. Softmax is applied column-wise, over context positions `t_z` for each query position `t_x`; therefore, the weights for each query sum to one. The output at query position `t_x` is the weighted sum of the value vectors. Since `exp(-inf) = 0`, masked positions receive zero weight.

GPT-2 Small self-attention shapes (`X = Z`, `l_x = l_z = l`):

```text
X, Z  : (768, l)
W_q   : (64, 768)      b_q : (64,)
W_k   : (64, 768)      b_k : (64,)
W_v   : (64, 768)      b_v : (64,)
Q, K  : (64, l)
V     : (64, l)
S     : (l, l)
Mask  : (l, l), upper-triangular Boolean matrix
V_tilde: (64, l)
```

## Algorithm 5 — Multi-head attention

**Input:**

- `X` in `R^(d_x x l_x)`: layer-normalized primary sequence
- `Z` in `R^(d_z x l_z)`: layer-normalized context sequence

**Output:** `V_tilde` in `R^(d_out x l_x)`

There are `H = 12` heads, each using the same mask and its own query, key, and value projections. For each head `h`:

```text
W_q_h in R^(d_attn x d_x)    b_q_h in R^d_attn
W_k_h in R^(d_attn x d_z)    b_k_h in R^d_attn
W_v_h in R^(d_mid  x d_z)    b_v_h in R^d_mid
```

The shared output projection has `W_o` in `R^(d_out x (H*d_mid))` and `b_o` in `R^d_out`.

```text
for h = 1, 2, ..., H:
    Y_h = ATTENTION(X, Z, W_qkv_h, Mask)  # (d_mid, l_x)

Y = concatenate(Y_1, Y_2, ..., Y_H, along feature/row axis)
V_tilde = W_o * Y + b_o * 1^T_lx
```

The head outputs are concatenated along the feature (row) dimension, not averaged. The output projection then mixes the concatenated head representations.

GPT-2 Small shapes:

```text
X, Z  : (768, l)
W_q_h : (64, 768)  per head h     b_q_h : (64,)
W_k_h : (64, 768)  per head h     b_k_h : (64,)
W_v_h : (64, 768)  per head h     b_v_h : (64,)
Y_h   : (64, l)    per head h
Y     : (768, l)   after concat (12 * 64 = 768)
W_o   : (768, 768)                 b_o   : (768,)
V_tilde: (768, l)
```

## Causal mask

For a sequence of length `l`, the mask has shape `(l, l)`. Rows are context positions `t_z`; columns are query positions `t_x`. A query can attend to the past and itself (`t_z <= t_x`).

Example for `l = 8`:

```text
        t_x:  1  2  3  4  5  6  7  8
t_z=1       [ 1  1  1  1  1  1  1  1 ]
t_z=2       [ 0  1  1  1  1  1  1  1 ]
t_z=3       [ 0  0  1  1  1  1  1  1 ]
t_z=4       [ 0  0  0  1  1  1  1  1 ]
t_z=5       [ 0  0  0  0  1  1  1  1 ]
t_z=6       [ 0  0  0  0  0  1  1  1 ]
t_z=7       [ 0  0  0  0  0  0  1  1 ]
t_z=8       [ 0  0  0  0  0  0  0  1 ]
```

In NumPy, because rows are `t_z` and columns are `t_x`, construct this mask with:

```python
mask = np.triu(np.ones((l, l), dtype=bool))
```

Future positions are masked by setting their scores to `-inf` before softmax.

## NumPy implementation notes

Broadcast a bias over sequence positions:

```python
b_q[:, np.newaxis]  # (d_attn, 1), broadcasts to (d_attn, l_x)
```

This is equivalent to `b_q * 1^T` in the pseudocode.

Numerically stable column-wise softmax:

```python
S_shifted = S - S.max(axis=0, keepdims=True)
A = np.exp(S_shifted) / np.exp(S_shifted).sum(axis=0, keepdims=True)
```

Apply the mask:

```python
S[~mask_bool] = -np.inf
```

Concatenate head outputs row-wise:

```python
Y = np.concatenate([Y_0, Y_1, ..., Y_11], axis=0)
# Shape: (12 * 64, l) = (768, l)
```

## Data flow — one GPT-2 self-attention layer

```text
Input X (768, l)
    |
    v
layer_norm -> X_norm (768, l)
    |
    v
MHATTENTION(X_norm, X_norm, W, Mask)
    |-- head 0:  (768, l) -> (64, l) --\
    |-- head 1:  (768, l) -> (64, l) --+-- concatenate along axis 0
    |        ...                       |
    |-- head 11: (768, l) -> (64, l) --/
    |                   Y (768, l)
    v
W_o * Y + b_o -> V_tilde (768, l)
    |
    v
Residual: X = X + V_tilde
```
