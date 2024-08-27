import numpy as np
import pandas as pd
from numpy.typing import NDArray


def f(X: NDArray) -> NDArray:
    return (X[:, 0] ** 2 - X[:, 1]) * np.sin(X[:, 2]) + X[:, 3]


rng = np.random.default_rng(1234)
X = rng.random([10, 4]) + 0.001
Y = f(X)
inputs = pd.DataFrame(X, columns=["v" + str(i) for i in range(4)])
inputs["sample id"] = np.arange(10)
inputs.to_csv("./example_in.csv", index=False)
target = pd.DataFrame(Y, columns=["target"])
target["sample id"] = np.arange(10)
target.to_csv("./example_out.csv", index=False)
inputs = pd.read_csv("~/Downloads/next_iteration.csv", index_col="sample id")
print(Y)
print(f(inputs.values))
