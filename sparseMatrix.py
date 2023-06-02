import random
import numpy as np
import scipy.sparse
from typing import List
import mpl_toolkits.axes_grid1
import matplotlib.pyplot as plt


def makA(n: int, d: int):
    # make a dense target matrix A
    A = np.random.random((n, d))
    return A


def makeS(n: int, r: int):
    # make a sparse embedding matrix S
    data = []
    rows = []
    cols = []
    for col in range(n):
        rows.append(random.randrange(r))
        cols.append(col)
        data.append(1 if random.random() < 0.5 else -1)
    S = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(r, n), dtype=np.float64)
    return S


def vis_At(A: np.ndarray):
    n, d = A.T.shape
    fig = plt.figure(figsize=(10, 0.5 + 10 * n / d))
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    im = ax.imshow(A.T, interpolation="none")
    fig.colorbar(im, cax=cax)
    fig.suptitle("A^T")
    plt.savefig("sparseMatrix_At.png")
    plt.show()


def vis_S(S: scipy.sparse.csr_matrix):
    arr = S.toarray()
    fig = plt.figure(figsize=(10, 0.5 + 10 * arr.shape[0] / arr.shape[1]))
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    im = ax.imshow(np.where(arr == 0, np.nan, arr), interpolation="none")
    fig.colorbar(im, cax=cax)
    fig.suptitle("S")
    plt.savefig("sparseMatrix_S.png")
    plt.show()


def vis_SA(S: scipy.sparse.csr_matrix, A: np.ndarray):
    arr = S.dot(A)
    fig = plt.figure(figsize=(5, -2 + 5 * arr.shape[0] / arr.shape[1]))
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    im = ax.imshow(np.where(arr == 0, np.nan, arr), interpolation="none")
    fig.colorbar(im, cax=cax)
    fig.suptitle("S*A")
    plt.savefig("sparseMatrix_SA.png")
    plt.show()


def main():
    n = 100
    d = 5
    r = 10
    A = makA(n, d)
    S = makeS(n, r)
    vis_At(A)
    vis_S(S)
    vis_SA(S, A)


if __name__ == "__main__":
    main()
