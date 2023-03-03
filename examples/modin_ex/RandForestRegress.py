from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import modin.pandas as pd

# import pandas as pd
import modin.config as cfg
import time

print("with modin")
print(cfg.NPartitions.get())  # prints '16'
X, y = make_regression(
    n_samples=100_000_000, n_features=4, n_informative=2, random_state=0, shuffle=False
)
start_time = time.time()
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)
print(X_df.shape)
regr = RandomForestRegressor(
    n_estimators=4,
    max_depth=2,
    # bootstrap=True,
    # max_samples=0.5,
    n_jobs=16,
    random_state=0,
)
regr.fit(X, y)

print(regr.predict([[0, 0, 0, 0]]))

exe_min, exe_sec = divmod(time.time() - start_time, 60)
print(f"finished in {exe_min} mins and {exe_sec} seconds")
