from tracemalloc import start
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

import modin.pandas as pd

# import pandas as pd
from multiprocessing import cpu_count
from modin.distributed.dataframe.pandas import unwrap_partitions
import modin.config as cfg
import sys
import time
import numpy as np

# import unidist.config as unidist_cfg

# cfg.Engine.put("unidist")
# cfg.StorageFormat.put("pandas")
# unidist_cfg.Backend.put("mpi")


# cfg.NPartitions.put(cpu_count())
start_time = time.time()
# print(cfg.NPartitions.get())  # prints '16'

# cfg.NPartitions.put(8)  # Changing value of `NPartitions`
print(cfg.NPartitions.get())  # prints '1'
print("with modin 100M")
X, y = make_blobs(
    n_samples=100_000_000,
    n_features=3,
    centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
    cluster_std=[0.2, 0.1, 0.2, 0.2],
    random_state=9,
)

# X = np.ones((500_000_000, 3))
# X = np.random.rand(3, 1000000)
print("got X, y")
X_df = pd.DataFrame(X, columns=["feat_A", "feat_B", "feat_C"])
exe_min, exe_sec = divmod(time.time() - start_time, 60)
print(f"pd.dataframe finished in {exe_min} mins and {exe_sec} seconds")

pca = PCA(n_components=3)
pca.fit(X_df)
print("var ratio: ", pca.explained_variance_ratio_)
print("var: ", pca.explained_variance_)

exe_min, exe_sec = divmod(time.time() - start_time, 60)
print(f"total finished in {exe_min} mins and {exe_sec} seconds")
