import numpy as np


def sketchingSVD(l: int, A: np.ndarray):
    N, M = A.shape
    assert l // 2 < M, "ell must not be greater than 2 * m"
    assert l < N, "ell must not be greater than n"
    B = np.zeros([l, M])

    zeroRows = np.where(np.isclose(np.sum(B, axis=1), 0.0, atol=1e-7))[0]
    zeroRowIdx = 0
    for i in range(N):
        B[zeroRows[zeroRowIdx]] = A[i]
        zeroRowIdx += 1
        if zeroRowIdx == len(zeroRows):
            U, SIGMA, V = np.linalg.svd(B, full_matrices=False)
            delta = SIGMA[l // 2] ** 2
            SIGMAtilda = np.sqrt(np.clip(SIGMA**2 - delta, 0.0, None))
            B = np.dot(np.diag(SIGMAtilda), V)
            zeroRows = np.where(np.isclose(np.sum(B, axis=1), 0.0, atol=1e-7))[0]
            zeroRowIdx = 0
    return B


def calculateError(A, B):
    ATA = np.dot(A.T, A)
    BTB = np.dot(B.T, B)
    return np.linalg.norm(ATA - BTB, ord=2)


def squaredFrobeniusNorm(A):
    return np.linalg.norm(A, ord="fro") ** 2
