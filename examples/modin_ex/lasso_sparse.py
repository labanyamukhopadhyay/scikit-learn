from time import time
from scipy import sparse
from scipy import linalg

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

import pandas as pd

X, y = make_regression(n_samples=20000, n_features=5000, random_state=0)
# create a copy of X in sparse format
# X_sp = sparse.coo_matrix(X)

# alpha = 1
# sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
# dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
# print(type(X_sp))

# X_df = pd.DataFrame(X)
# y_df = pd.DataFrame(y)
# t0 = time()
# sparse_lasso.fit(X_sp, y)
# print(f"Sparse Lasso done in {(time() - t0):.3f}s")

# t0 = time()
# dense_lasso.fit(X, y)
# print(f"Dense Lasso done in {(time() - t0):.3f}s")

# # compare the regression coefficients
# coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
# print(f"Distance between coefficients : {coeff_diff:.2e}")

# make a copy of the previous data
Xs = X.copy()
# make Xs sparse by replacing the values lower than 2.5 with 0s
Xs[Xs < 2.5] = 0.0
# create a copy of Xs in sparse format
# Xs_sp = sparse.coo_matrix(Xs)
# Xs_sp = Xs_sp.tocsc()

# compute the proportion of non-zero coefficient in the data matrix
# print(f"Matrix density : {(Xs_sp.nnz / float(X.size) * 100):.3f}%")

alpha = 0.1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

# t0 = time()
# sparse_lasso.fit(Xs_sp, y)
# print(f"Sparse Lasso done in {(time() - t0):.3f}s")

X_df = pd.DataFrame(Xs)
y_df = pd.DataFrame(y)

t0 = time()
# dense_lasso.fit(Xs, y)
dense_lasso.fit(X_df, y_df)
print(f"Dense Lasso done in  {(time() - t0):.3f}s")

# # compare the regression coefficients
# coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
# print(f"Distance between coefficients : {coeff_diff:.2e}")
