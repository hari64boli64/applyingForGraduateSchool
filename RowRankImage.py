import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 画像の読み込み
img = Image.open("i_u_tokyo.jpg")
img = img.convert("L")

# 特異値分解
U, S, V = np.linalg.svd(img)

fig = plt.figure(figsize=(15, 15 * img.size[1] / img.size[0]))

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["black", "#539BB2", "white"]
)

# オリジナル画像の表示
ax0 = fig.add_subplot(3, 3, 1)
img0 = ax0.imshow(img, cmap=cmap)
ax0.set_title(f"Original Image (total rank:{len(S)})")
fig.colorbar(img0, ax=ax0)

# 再構成画像の表示
for idx, k in enumerate([2**i for i in range(9 - 1)]):
    ax = fig.add_subplot(3, 3, idx + 2)
    rank_k = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
    img = ax.imshow(rank_k, cmap=cmap)
    ax.set_title(f"Row Rank Approximation (rank:{k})")
    fig.colorbar(img, ax=ax)

# 画像の保存
fig.suptitle("Row Rank Approximation with SVD")
plt.tight_layout()
plt.savefig("RowRankImage.png")
plt.show()
