import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing.preprocessing import supervised_preparation
from src.utils.visualizations import plot_xy_results
warnings.filterwarnings("ignore")

# read data
path_cleaned = "data/cleaned-data.csv"
df = pd.read_csv(path_cleaned)
df.drop(columns=["fecha"], inplace=True)

# target column
target = ['%cu_conc_final']

# separacion target features
columns = list(df.columns)
for col in target:
    columns.remove(col)

# target y features
y = df[target]
x = df[columns]

# división del conjutno de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=21)

# supervised learning
x_train, y_train = supervised_preparation(x_train, y_train)
x_test, y_test = supervised_preparation(x_test, y_test)

# data normalization
sc = MinMaxScaler(feature_range=(0, 1))
# training
x_train = sc.fit_transform(x_train)
# testing
x_test = sc.transform(x_test)

# fiting the model
reg = RandomForestRegressor(n_estimators=20, random_state=21)
reg.fit(x_train, y_train)

score = reg.score(x_test, y_test)
predictions = reg.predict(x_test)
predictions = np.reshape(predictions, (-1, 1))

plot_xy_results(predictions, y_test, index=str(1), name=target[0],
                folder_name="random-forest")
