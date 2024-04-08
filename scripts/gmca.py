import numpy as np

def mad(xin = 0):
	return np.median(abs(xin - np.median(xin)))/0.6735

def soft_threshold(x, level):
    return np.sign(x) * np.maximum(np.abs(x) - level, 0.)

def GMCA(X, n_sources, max_iter=100):
    A = np.random.randn(X.shape[0], n_sources)
    for _ in range(max_iter):
        S = np.linalg.pinv(A) @ X
        for SS in S:
            level = 3.0 * mad(SS)
            SS = soft_threshold(SS, level)
        A = X @ S.T @ np.linalg.inv(S @ S.T)
        A = A / np.sqrt(np.sum(A**2, axis=1, keepdims=True))
        A = np.maximum(A, 0)
    return A, S