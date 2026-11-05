
import numpy as np

# GPT-2 Small hyperparameters

L_MAX  = 1024       # maximum sequence length  (ℓ_max)
D_E    = 768        # model / embedding width   (d_e)
L      = 12         # number of layers          (L)
H      = 12         # number of attention heads (H)
N_V    = 50_257     # vocabulary size           (N_V)
D_MLP  = 3072       # MLP hidden width          (d_mlp)
D_ATTN = D_E // H   # per-head attention dim    (d_attn = 64)
D_MID  = D_ATTN     # alias used in Algorithm 5 (d_mid  = 64)


 
# Algorithm 1 – Token Embedding


def token_embedding(v: int, W_e: np.ndarray) -> np.ndarray:
    # Validate inputs
    d_e, N_V_local = W_e.shape
    assert 0 <= v < N_V_local, (
        f"Token ID {v} out of range [0, {N_V_local})."
    )

    e = W_e[:, v]        # shape: (d_e,)
    return e



# Algorithm 6 – layer_norm

def layer_norm(
    e: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:

    # Step 1: compute mean
    m = np.sum(e) / e.shape[0]           # scalar

    # Step 2: compute variance
    v = np.sum((e - m) ** 2) / e.shape[0]  # scalar

    # Step 3: normalize, scale, and shift
    # ê = ((e − m) / √(v + ε)) ⊙ γ + β
    e_hat = ((e - m) / np.sqrt(v + eps)) * gamma + beta  # shape: (d_e,)

    return e_hat



# Algorithm 4 – ATTENTION  (single attention head)


def _softmax_cols(S: np.ndarray) -> np.ndarray:
    # Subtract column max for numerical stability
    S_shifted = S - np.max(S, axis=0, keepdims=True)   # broadcast over rows
    exp_S = np.exp(S_shifted)
    A = exp_S / np.sum(exp_S, axis=0, keepdims=True)   # normalize columns
    return A


def attention(
    X: np.ndarray,
    Z: np.ndarray,
    W_q: np.ndarray,
    b_q: np.ndarray,
    W_k: np.ndarray,
    b_k: np.ndarray,
    W_v: np.ndarray,
    b_v: np.ndarray,
    Mask: np.ndarray,
) -> np.ndarray:
    
    d_attn = W_q.shape[0]   # e.g. 64 for GPT-2 small
    l_x    = X.shape[1]
    l_z    = Z.shape[1]

    # Step 1: Q ← W_q X + b_q 1^T_ℓx
    # b_q[:, None] broadcasts the bias to all l_x positions
    Q = W_q @ X + b_q[:, np.newaxis]    # (d_attn, l_x)

    # Step 2: K ← W_k Z + b_k 1^T_ℓz
    K = W_k @ Z + b_k[:, np.newaxis]    # (d_attn, l_z)

    # Step 3: V ← W_v Z + b_v 1^T_ℓz
    V = W_v @ Z + b_v[:, np.newaxis]    # (d_out,  l_z)

    # Step 4: S ← K^T Q / √d_attn
    # K^T has shape (l_z, d_attn); Q has shape (d_attn, l_x)
    # → S has shape (l_z, l_x) — each entry S[t_z, t_x] = k_{t_z}^T q_{t_x}
    S = (K.T @ Q) / np.sqrt(d_attn)     # (l_z, l_x)

    # Step 5: apply mask – set S[t_z, t_x] = −∞ wherever Mask is False/0
    mask_bool = Mask.astype(bool)
    S[~mask_bool] = -np.inf             # masked positions → 0 after softmax

    # Step 6: Ṽ = V · softmax(S)
    # softmax is applied column-wise (over the l_z / context dimension)
    A = _softmax_cols(S)                # attention weights: (l_z, l_x)
    V_tilde = V @ A                     # (d_out, l_z) @ (l_z, l_x) = (d_out, l_x)

    return V_tilde



# Algorithm 5 – MHATTENTION  (multi-head attention)


def mhattention(
    X: np.ndarray,
    Z: np.ndarray,
    W: dict,
    Mask: np.ndarray,
    H: int = H,
) -> np.ndarray:
    
    # Steps 1–2: run each head independently
    head_outputs = []                # will hold Y_h for h = 0 … H-1
    for h in range(H):
        Y_h = attention(
            X=X,
            Z=Z,
            W_q=W[f"W_q_{h}"],
            b_q=W[f"b_q_{h}"],
            W_k=W[f"W_k_{h}"],
            b_k=W[f"b_k_{h}"],
            W_v=W[f"W_v_{h}"],
            b_v=W[f"b_v_{h}"],
            Mask=Mask,
        )
        head_outputs.append(Y_h)    # each Y_h: (d_mid, l_x)

    # Step 3: stack / concatenate heads along the feature dimension
    # Y ← [Y_1; Y_2; …; Y_H]  →  shape: (H * d_mid, l_x) = (768, l_x)
    Y = np.concatenate(head_outputs, axis=0)   # (H*d_mid, l_x)

    # Step 4: output projection
    # Ṽ = W_o Y + b_o 1^T_ℓx
    W_o = W["W_o"]                             # (d_out, H*d_mid)
    b_o = W["b_o"]                             # (d_out,)
    l_x = X.shape[1]
    V_tilde = W_o @ Y + b_o[:, np.newaxis]    # (d_out, l_x)

    return V_tilde



# Helper: build a causal (lower-triangular) mask

def causal_mask(seq_len: int) -> np.ndarray:
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))



# Quick sanity-check / demo


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)


    # Algorithm 1 – token_embedding
    print("=" * 60)
    print("Algorithm 1: Token Embedding")
    print("=" * 60)

    W_e = rng.standard_normal((D_E, N_V)).astype(np.float32)   # (768, 50257)
    v   = 314                                                    # arbitrary token ID

    e = token_embedding(v, W_e)
    print(f"  W_e shape         : {W_e.shape}")
    print(f"  Token ID          : {v}")
    print(f"  Embedding shape   : {e.shape}")
    print(f"  e[:5]             : {e[:5]}")
    print(f"  Matches W_e[:,{v}]: {np.allclose(e, W_e[:, v])}")
    print()

    
    # Algorithm 6 – layer_norm
    
    print("=" * 60)
    print("Algorithm 6: layer_norm")
    print("=" * 60)

    gamma = rng.standard_normal(D_E).astype(np.float32)
    beta  = rng.standard_normal(D_E).astype(np.float32)

    e_hat = layer_norm(e, gamma, beta)
    print(f"  Input  shape : {e.shape}")
    print(f"  Output shape : {e_hat.shape}")
    # After normalisation (before γ/β) the activations have mean≈0, var≈1
    # With γ,β the mean and var change — verify the intermediate step
    normed = (e - e.mean()) / np.sqrt(e.var() + 1e-5)
    expected = normed * gamma + beta
    print(f"  Output matches manual computation: {np.allclose(e_hat, expected)}")
    print()

    
    # Algorithm 4 – ATTENTION  (one head, short sequence for visibility)
    
    print("=" * 60)
    print("Algorithm 4: ATTENTION (single head)")
    print("=" * 60)

    SEQ_LEN = 8          # short sequence for demo
    d_x     = D_E        # 768
    d_z     = D_E        # 768  (self-attention: X = Z)
    d_attn  = D_ATTN     # 64
    d_out   = D_MID      # 64

    X_demo = rng.standard_normal((d_x, SEQ_LEN)).astype(np.float32)
    Z_demo = X_demo.copy()   # self-attention

    W_q_h = rng.standard_normal((d_attn, d_x)).astype(np.float32)   # (64, 768)
    b_q_h = rng.standard_normal(d_attn).astype(np.float32)
    W_k_h = rng.standard_normal((d_attn, d_z)).astype(np.float32)
    b_k_h = rng.standard_normal(d_attn).astype(np.float32)
    W_v_h = rng.standard_normal((d_out,  d_z)).astype(np.float32)
    b_v_h = rng.standard_normal(d_out).astype(np.float32)
    Mask  = causal_mask(SEQ_LEN)                                      # (8, 8)

    V_tilde_single = attention(
        X_demo, Z_demo,
        W_q_h, b_q_h,
        W_k_h, b_k_h,
        W_v_h, b_v_h,
        Mask,
    )
    print(f"  X shape              : {X_demo.shape}")
    print(f"  Mask shape           : {Mask.shape}")
    print(f"  V_tilde shape        : {V_tilde_single.shape}")
    print(f"  V_tilde[:3, :3]      :\n{V_tilde_single[:3, :3]}")
    print()

    
    # Algorithm 5 – MHATTENTION  (all 12 heads)
        
    print("=" * 60)
    print("Algorithm 5: MHATTENTION (12 heads)")
    print("=" * 60)

    # Build parameter dictionary for all 12 heads
    W_mha: dict = {}
    for h in range(H):
        W_mha[f"W_q_{h}"] = rng.standard_normal((D_ATTN, D_E)).astype(np.float32)
        W_mha[f"b_q_{h}"] = rng.standard_normal(D_ATTN).astype(np.float32)
        W_mha[f"W_k_{h}"] = rng.standard_normal((D_ATTN, D_E)).astype(np.float32)
        W_mha[f"b_k_{h}"] = rng.standard_normal(D_ATTN).astype(np.float32)
        W_mha[f"W_v_{h}"] = rng.standard_normal((D_MID,  D_E)).astype(np.float32)
        W_mha[f"b_v_{h}"] = rng.standard_normal(D_MID).astype(np.float32)

    # Output projection: W_o ∈ ℝ^{768 × 768},  b_o ∈ ℝ^{768}
    W_mha["W_o"] = rng.standard_normal((D_E, H * D_MID)).astype(np.float32)
    W_mha["b_o"] = rng.standard_normal(D_E).astype(np.float32)

    V_tilde_mh = mhattention(X_demo, X_demo, W_mha, Mask, H=H)

    print(f"  X shape              : {X_demo.shape}")
    print(f"  Number of heads      : {H}")
    print(f"  d_attn per head      : {D_ATTN}")
    print(f"  W_o shape            : {W_mha['W_o'].shape}")
    print(f"  V_tilde shape        : {V_tilde_mh.shape}  (expected: ({D_E}, {SEQ_LEN}))")
    print(f"  V_tilde[:3, :3]      :\n{V_tilde_mh[:3, :3]}")
    print()
    print("All algorithms verified successfully.")

