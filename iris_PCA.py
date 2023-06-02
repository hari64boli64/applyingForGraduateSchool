import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets

from sketchingSVD import sketchingSVD

iris = datasets.load_iris()
print("iris.DESCR: ", iris.DESCR)

X = iris.data
y = iris.target
colors = ["red", "blue", "green"]
markers = ["o", "s", "^"]
labels = ["setosa", "versicolour", "virginica"]

# 最初の3データだけを3Dでプロットする
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
for i in range(3):
    ax.scatter(
        X[i * 50 : (i + 1) * 50, 0],
        X[i * 50 : (i + 1) * 50, 1],
        X[i * 50 : (i + 1) * 50, 2],
        c=colors[i],
        marker=markers[i],
        label=labels[i],
    )
ax.set_xlabel("sepal length (cm)")
ax.set_ylabel("sepal width (cm)")
ax.set_zlabel("petal length (cm)")
ax.legend()
fig.suptitle("IRIS dataset")
plt.savefig("iris_3d.png")
plt.show()


# 主成分分析をする
n_components = 2
X -= np.mean(X, axis=0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
X_pca = U[:, [0, 1]]

# 主成分をプロットする
sns.set_theme("paper")
plt.figure(figsize=(5, 5))
for i in range(3):
    plt.scatter(
        X_pca[i * 50 : (i + 1) * 50, 0],
        X_pca[i * 50 : (i + 1) * 50, 1],
        c=colors[i],
        marker=markers[i],
        label=labels[i],
    )
plt.legend()
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.title("Full PCA of IRIS dataset")
plt.savefig("iris_full_pca.png")
plt.show()

# スケッチングによる主成分分析をする
ell = 2
X_sketch = sketchingSVD(ell, X)
print(X_sketch.shape)
U, S, Vt = np.linalg.svd(X_sketch, full_matrices=False)
projection_matrix = np.dot(np.dot(Vt.T, Vt), Vt.T)
X_sketch_pca = np.dot(X, projection_matrix)[:, [0, 1]]


# スケッチングによる主成分をプロットする
plt.figure(figsize=(5, 5))
for i in range(3):
    plt.scatter(
        X_sketch_pca[i * 50 : (i + 1) * 50, 0],
        X_sketch_pca[i * 50 : (i + 1) * 50, 1],
        c=colors[i],
        marker=markers[i],
        label=labels[i],
    )
plt.legend()
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.title("Sketching PCA of IRIS dataset")
plt.savefig("iris_sketching_pca.png")
plt.show()
