@echo off
chcp 65001 >nul
cls

echo ============================================================
echo   GPT-2 Small  --  Formal Algorithms Reference Sheet
echo   Source: "Formal Algorithms for Transformers"
echo   Phuong and Hutter, DeepMind (2022) arXiv:2207.09238
echo ============================================================
echo.

echo ============================================================
echo  SECTION 2  --  GPT-2 Small Hyperparameters
echo ============================================================
echo.
echo   Symbol      Value   Meaning
echo   --------    -----   -----------------------------------
echo   l_max       1024    Maximum sequence length
echo   d_e          768    Embedding / model width
echo   L             12    Number of transformer layers
echo   H             12    Number of attention heads
echo   N_V       50,257    Vocabulary size
echo   d_mlp       3072    MLP hidden width  (= 4 * d_e)
echo   d_attn        64    Per-head attention dim  (= d_e / H)
echo   d_mid         64    Per-head value dim  (alias of d_attn)
echo.
echo   Derived relation:
echo     d_attn = d_mid = d_e / H = 768 / 12 = 64
echo     H * d_mid = 12 * 64 = 768 = d_e   (heads re-join model width)
echo.

echo ============================================================
echo  ALGORITHM 1  --  Token Embedding
echo ============================================================
echo.
echo   Input  : v  in  V  [N_V]          -- token ID (integer)
echo   Output : e  in  R^d_e             -- vector for token v
echo   Params : W_e  in  R^(d_e x N_V)  -- token embedding matrix
echo.
echo   Pseudocode:
echo     1:  return  e = W_e[:, v]
echo.
echo   Explanation:
echo     W_e stores one d_e-dim column per vocabulary token.
echo     Selecting column v directly gives the embedding vector.
echo     (Column-vector convention: tokens are columns of W_e.)
echo.
echo   GPT-2 Small shapes:
echo     W_e  :  (768, 50257)
echo     v    :  integer in [0, 50256]
echo     e    :  (768,)
echo.

echo ============================================================
echo  ALGORITHM 6  --  layer_norm
echo ============================================================
echo.
echo   Input  : e         in  R^d_e   -- neural network activations
echo   Output : e_hat     in  R^d_e   -- normalized activations
echo   Params : gamma, beta  in R^d_e -- scale and offset (learned)
echo.
echo   Pseudocode:
echo     1:  m  :=  (1/d_e) * SUM_{i=1}^{d_e}  e[i]              (mean)
echo     2:  v  :=  (1/d_e) * SUM_{i=1}^{d_e}  (e[i] - m)^2     (variance)
echo     3:  return  e_hat = ( (e - m) / sqrt(v + eps) ) (*) gamma + beta
echo.
echo   Notes:
echo     (*) denotes element-wise (Hadamard) multiplication
echo     eps = 1e-5  added for numerical stability
echo     m   is a scalar (mean over all d_e elements)
echo     v   is a scalar (variance over all d_e elements)
echo.
echo   NumPy mapping:
echo     m     = np.sum(e) / e.shape[0]
echo     v     = np.sum((e - m)**2) / e.shape[0]
echo     e_hat = ((e - m) / np.sqrt(v + eps)) * gamma + beta
echo.

echo ============================================================
echo  ALGORITHM 4  --  ATTENTION  (single head)
echo ============================================================
echo.
echo   Input  : X  in R^(d_x  x l_x)  -- primary sequence  (queries from X)
echo            Z  in R^(d_z  x l_z)  -- context sequence  (keys/values from Z)
echo   Output : V~ in R^(d_out x l_x) -- updated token representations
echo.
echo   Parameters  W_qkv:
echo     W_q in R^(d_attn x d_x),   b_q in R^d_attn   (query projection)
echo     W_k in R^(d_attn x d_z),   b_k in R^d_attn   (key   projection)
echo     W_v in R^(d_out  x d_z),   b_v in R^d_out    (value projection)
echo.
echo   Hyperparameters:
echo     Mask in {0,1}^(l_z x l_x)
echo     Mask[t_z, t_x] = 1  implies query t_x may attend to context t_z
echo     Mask[t_z, t_x] = 0  implies score forced to -inf before softmax
echo     Causal mask (GPT-2): Mask[t_z, t_x] = 1  iff  t_z ^<= t_x
echo.
echo   Pseudocode:
echo     1:  Q  :=  W_q * X + b_q * 1^T_lx      [Query  : d_attn x l_x]
echo     2:  K  :=  W_k * Z + b_k * 1^T_lz      [Key    : d_attn x l_z]
echo     3:  V  :=  W_v * Z + b_v * 1^T_lz      [Value  : d_out  x l_z]
echo     4:  S  :=  K^T * Q / sqrt(d_attn)       [Score  : l_z x l_x]
echo     5:  for all t_z, t_x:
echo             if NOT Mask[t_z, t_x]  then  S[t_z, t_x] = -inf
echo     6:  return  V~ = V * softmax_colwise(S)  [V~     : d_out x l_x]
echo.
echo   Notes:
echo     "1^T" is a row-vector of ones -- broadcasts bias over all positions
echo     S = K^T * Q  has shape (l_z, l_x); each entry is k_{t_z}^T * q_{t_x}
echo     softmax is applied COLUMN-WISE (over t_z for each fixed t_x)
echo       so weights for each query sum to 1 over all context positions
echo     V~ col t_x = SUM_{t_z} softmax(S)[t_z, t_x] * V[:, t_z]
echo       (weighted average of value vectors under attention weights)
echo     exp(-inf) = 0, so masked positions receive exactly zero weight
echo.
echo   GPT-2 Small shapes (self-attention: X = Z, l_x = l_z = l):
echo     X, Z   :  (768, l)
echo     W_q    :  (64,  768)      b_q  :  (64,)
echo     W_k    :  (64,  768)      b_k  :  (64,)
echo     W_v    :  (64,  768)      b_v  :  (64,)
echo     Q      :  (64,  l)
echo     K      :  (64,  l)
echo     V      :  (64,  l)
echo     S      :  (l,   l)
echo     Mask   :  (l,   l)   lower-triangular bool matrix
echo     V~     :  (64,  l)
echo.

echo ============================================================
echo  ALGORITHM 5  --  MHATTENTION  (multi-head attention)
echo ============================================================
echo.
echo   Input  : X  in R^(d_x  x l_x)   -- layer-normalised primary sequence
echo            Z  in R^(d_z  x l_z)   -- layer-normalised context sequence
echo   Output : V~ in R^(d_out x l_x)  -- multi-head attention output
echo.
echo   Hyperparameters:
echo     H    = 12                       number of attention heads
echo     Mask in {0,1}^(l_z x l_x)     (same mask used by every head)
echo.
echo   Parameters  W:
echo     For each head h in [H] = [12]:
echo       W_q_h  in R^(d_attn x d_x)    b_q_h  in R^d_attn
echo       W_k_h  in R^(d_attn x d_z)    b_k_h  in R^d_attn
echo       W_v_h  in R^(d_mid  x d_z)    b_v_h  in R^d_mid
echo     Output projection (shared):
echo       W_o   in R^(d_out x H*d_mid)  b_o   in R^d_out
echo.
echo   Pseudocode:
echo     1:  for h = 1, 2, ..., H  do
echo     2:      Y_h  :=  ATTENTION(X, Z, W_qkv_h, Mask)    [d_mid x l_x]
echo     3:  Y  :=  [Y_1 ; Y_2 ; ... ; Y_H]                 [H*d_mid x l_x]
echo     4:  return  V~ = W_o * Y + b_o * 1^T_lx            [d_out x l_x]
echo.
echo   Notes:
echo     Step 2: each head runs INDEPENDENTLY with its own W_q, W_k, W_v
echo     Step 3: outputs are CONCATENATED along the feature (row) dimension
echo             NOT averaged -- each head's 64-dim block occupies its own rows
echo     Step 4: W_o mixes all heads together into a single d_out-dim output
echo.
echo   GPT-2 Small shapes:
echo     X, Z     :  (768, l)
echo     W_q_h    :  (64,  768)  per head h    b_q_h  :  (64,)
echo     W_k_h    :  (64,  768)  per head h    b_k_h  :  (64,)
echo     W_v_h    :  (64,  768)  per head h    b_v_h  :  (64,)
echo     Y_h      :  (64,  l)    per head h
echo     Y        :  (768, l)    after concat  [12 * 64 = 768]
echo     W_o      :  (768, 768)                b_o    :  (768,)
echo     V~       :  (768, l)
echo.

echo ============================================================
echo  CAUSAL MASK  --  Used in GPT-2 Self-Attention
echo ============================================================
echo.
echo   Shape  : (l, l)  where l = sequence length
echo   Rule   : Mask[t_z, t_x] = 1  iff  t_z ^<= t_x
echo.
echo   Example for l = 8  (rows = t_z, cols = t_x):
echo.
echo         t_x:  1  2  3  4  5  6  7  8
echo     t_z=1  [  1  1  1  1  1  1  1  1 ]
echo     t_z=2  [  0  1  1  1  1  1  1  1 ]
echo     t_z=3  [  0  0  1  1  1  1  1  1 ]
echo     t_z=4  [  0  0  0  1  1  1  1  1 ]
echo     t_z=5  [  0  0  0  0  1  1  1  1 ]
echo     t_z=6  [  0  0  0  0  0  1  1  1 ]
echo     t_z=7  [  0  0  0  0  0  0  1  1 ]
echo     t_z=8  [  0  0  0  0  0  0  0  1 ]
echo.
echo   Column t_x can only attend to rows t_z ^<= t_x (past + self).
echo   Future positions are masked out (set to -inf before softmax).
echo   NumPy: np.tril(np.ones((l, l), dtype=bool))
echo.

echo ============================================================
echo  NUMPY IMPLEMENTATION NOTES
echo ============================================================
echo.
echo   Broadcasting bias over sequence positions:
echo     b_q[:, np.newaxis]  has shape (d_attn, 1) -- broadcasts to (d_attn, l_x)
echo     equivalent to  b_q * 1^T  in the pseudocode
echo.
echo   Column-wise softmax (numerically stable):
echo     S_shifted = S - S.max(axis=0, keepdims=True)   -- subtract col max
echo     A = exp(S_shifted) / exp(S_shifted).sum(axis=0, keepdims=True)
echo.
echo   Applying the mask:
echo     S[~mask_bool] = -np.inf   -- masked entries become 0 after softmax
echo.
echo   Head concatenation:
echo     np.concatenate([Y_0, Y_1, ..., Y_11], axis=0)  -- stack row-wise
echo     result shape: (12*64, l) = (768, l)
echo.

echo ============================================================
echo  DATA FLOW SUMMARY  (one transformer layer, GPT-2 self-attn)
echo ============================================================
echo.
echo   Input X  (768, l)
echo       ^|
echo       v
echo   layer_norm  --^>  X_norm  (768, l)
echo       ^|
echo       v
echo   MHATTENTION(X_norm, X_norm, W, Mask)
echo       ^|-- head 0: ATTENTION  (768,l) --^> (64,l)  ---\
echo       ^|-- head 1: ATTENTION  (768,l) --^> (64,l)  -- concat axis=0
echo       ^|   ...                                      --/
echo       ^|-- head 11: ATTENTION (768,l) --^> (64,l)  --/
echo       ^|                                             ^|
echo       ^|           Y = [Y_0;...;Y_11]  (768, l) ^<---/
echo       ^|
echo       v
echo   W_o * Y + b_o  --^>  V_tilde  (768, l)
echo       ^|
echo       v
echo   Residual: X = X + V_tilde
echo.

echo ============================================================
echo   End of Reference Sheet
echo ============================================================
echo.
pause
