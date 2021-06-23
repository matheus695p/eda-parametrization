import pandas as pd
from src.preprocessing.correlations import get_columns_correlated
from src.utils.visualizations import correlation_matrix

# do a quick scan with pandas profiling, given the nature of
# the proof
path_featured = "data/featured_data.csv"
df = pd.read_csv(path_featured)

# just get the numerical columns
numerical_cols = []
for col in df.columns:
    type_ = str(df[[col]].dtypes[0])
    if (type_ == "int64") | (type_ == "float64"):
        print(col)
        numerical_cols.append(col)

df = df[numerical_cols]

# get correlations relationships
pearson = get_columns_correlated(df, method="pearson", threshold=0.8)
spearman = get_columns_correlated(df, method="spearman", threshold=0.8)
kendall = get_columns_correlated(df, method="pearson", threshold=0.8)

# visualize correlations
correlation_matrix(df, method="pearson")
correlation_matrix(df, method="spearman")
correlation_matrix(df, method="kendall")

# I have to drop the column %feconcentrado_limpieza_rougher because is too
# correlated with other features
