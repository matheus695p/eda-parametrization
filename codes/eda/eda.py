import pandas as pd
from src.eda.exploratory_data_analysis import profiling_report

# do a quick scan with pandas profiling, given the nature of
# the proof
path = "data/dataFlotacion.csv"
df = pd.read_csv(path)
profiling_report(df, minimal_mode=True, dark_mode=True)
# after seeing the .html, there is a correct data imputation, but some columns
# need to be eliminated
# let's see how the correlation matrices behaved


# run code codes/cleaning/clean.py to get the next file
path_cleaned = "data/cleaned-data.csv"
df = pd.read_csv(path_cleaned)
profiling_report(df, minimal_mode=True, dark_mode=True)
