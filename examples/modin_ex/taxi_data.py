from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import modin.pandas as pd
import modin.config as cfg
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(
    "examples/modin_ex/yellow_tripdata_2015-01.csv",
    header=0,
)
df = df.drop(
    [
        "TPEP_PICKUP_DATETIME",
        "TPEP_DROPOFF_DATETIME",
        "CONGESTION_SURCHARGE",
        "AIRPORT_FEE",
        " ",
    ],
    1,
)

print(df.isna().sum())
le = LabelEncoder()
df["STORE_AND_FWD_FLAG"] = le.fit_transform(df["STORE_AND_FWD_FLAG"])

x = df.drop("TIP_AMOUNT", 1)
y = df.TIP_AMOUNT


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

lr = LinearRegression()
dt = DecisionTreeRegressor()
rn = RandomForestRegressor(n_jobs=8)
knn = KNeighborsRegressor()
sgd = SGDRegressor()
br = BaggingRegressor()

li = [lr, knn, rn, dt, br]
di = {}
for i in li:
    i.fit(xtrain, ytrain)
    ypred = i.predict(xtest)
    print(i, ":", r2_score(ypred, ytest) * 100)
    di.update({str(i): i.score(xtest, ytest) * 100})

plt.figure(figsize=(15, 6))
plt.title("Algorithm vs Accuracy", fontweight="bold")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(
    di.keys(),
    di.values(),
    marker="o",
    color="plum",
    linewidth=4,
    markersize=13,
    markerfacecolor="gold",
    markeredgecolor="slategray",
)
plt.show()
