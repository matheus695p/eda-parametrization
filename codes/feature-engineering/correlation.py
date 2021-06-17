import pandas as pd
from src.utils.visualizations import (correlation_matrix)

# do a quick scan with pandas profiling, given the nature of
# the proof
path_cleaned = "data/cleaned-data.csv"
df = pd.read_csv(path_cleaned)

# just get the numerical columns
numerical_cols = []
for col in df.columns:
    type_ = str(df[[col]].dtypes[0])
    if (type_ == "int64") | (type_ == "float64"):
        print(col)
        numerical_cols.append(col)

df = df[numerical_cols]

# ver correlaciones
correlation_matrix(df, method="pearson")
correlation_matrix(df, method="spearman")
correlation_matrix(df, method="kendall")
