import numpy as np

def attention(X, W):
    Wq, Wk, Wv, Bq, Bk, Bv = W

    # Calculate Query, Key and Value
    Q = Wq @ X + Bq[:, None]
    K = Wk @ X + Bk[:, None]
    V = Wv @ X + Bv[:, None]

    # Calculate attention scores
    S = (K.T @ Q) / np.sqrt(Wq.shape[0])

    # Apply causal mask
    mask = np.triu(np.ones(S.shape, dtype=bool), k=1)
    S = np.where(mask, -np.inf, S)

    # Calculate softmax
    S = S - np.max(S, axis=0, keepdims=True)
    A = np.exp(S)
    A = A / np.sum(A, axis=0, keepdims=True)

    # Calculate output
    Y = V @ A

    return Y


def test_attention():
    X = np.array([[1., 2., 3.],
                  [4., 5., 6.]])

    Wq = np.array([[1., 0.],
                   [0., 1.]])

    Wk = np.array([[1., 0.],
                   [0., 1.]])

    Wv = np.array([[1., 0.],
                   [0., 1.]])

    Bq = np.zeros(2)
    Bk = np.zeros(2)
    Bv = np.zeros(2)

    W = (Wq, Wk, Wv, Bq, Bk, Bv)

    Y = attention(X, W)

    print("Attention output:")
    print(Y)
    print("Output shape:", Y.shape)


test_attention()