import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from src.preprocessing.feature import (exponential_features,
                                       polinomial_features,
                                       log_features, selection_by_correlation)
plt.style.use('dark_background')

path_cleaned = "data/cleaned-data.csv"
date_format = "%Y-%m-%d %H:%M:%S"

# Read data
df = pd.read_csv(path_cleaned)
df["fecha"] = df["fecha"].apply(lambda x: datetime.strptime(x, date_format))
dates = df[["fecha"]]
df.set_index(["fecha"], inplace=True)

# pairtplot to see distributions
# sns.pairplot(df)
# columns to featured
columns = list(df.columns)

# columnas de predicción
targets = ['%cu_conc_final']
for col in targets:
    columns.remove(col)

# exponential
df = exponential_features(df, columns)
# logarithm
df = log_features(df, columns)
# aqrt
df = polinomial_features(df, columns, grade=0.5)
# polinomial 2
df = polinomial_features(df, columns, grade=2)
# selecting by correlation
corr = selection_by_correlation(df, threshold=0.8)
corr.replace([np.inf, -np.inf], np.nan, inplace=True)

# check correlation matrix
corr_matrix = corr.corr()

# droping columns with nans
nans = pd.DataFrame(corr.isna().sum(), columns=["counter"])
nans.reset_index(drop=False, inplace=True)
nans.rename(columns={"index": "column"}, inplace=True)
nans["percentage"] = nans["counter"] / len(corr) * 100
droping_cols = nans[nans["percentage"] > 0]["column"].to_list()
# droping_cols.append("index")
corr.drop(columns=droping_cols, inplace=True)
# concat date
corr = pd.concat([dates, corr], axis=1)
# path for regression
corr.reset_index(drop=True, inplace=True)
path_regression = "data/featured_data.csv"
corr.to_csv(path_regression, index=False)
