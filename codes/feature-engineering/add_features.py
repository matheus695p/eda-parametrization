import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')

path_cleaned = "data/cleaned-data.csv"
df = pd.read_csv(path_cleaned)


# replacing this columns by his logaritm(x + 1)

for col in df.columns:
    print(col)
    if "fecha" == col:
        pass
    else:
        new_col = col + "_log"
        df[new_col] = df[col].apply(lambda x: np.log(x + 1))
        sns.displot(df[[col, new_col]])


# im going to use the results from the exploratory analysis to add more
# features
# columns skewed
columns_log = ["xantato", "nahs", "%fe_alimen._rougher", "%cu_colas_rougher"]
