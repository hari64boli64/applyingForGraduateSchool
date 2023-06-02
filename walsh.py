import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def sylvester(n: int):
    if n == 1:
        return np.array([[1, 1], [1, -1]])
    else:
        return np.kron(sylvester(n - 1), sylvester(1))


H3 = sylvester(3)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FF0000", "#00FF00"])

plt.figure(figsize=(5, 5))
plt.imshow(H3, cmap=cmap)
plt.title("Walsh matrix (N=3)")
plt.savefig("walsh.png")
plt.show()
