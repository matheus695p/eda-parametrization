import pandas as pd
from autoplotter import run_app

path = "data/dataFlotacion.csv"
df = pd.read_csv(path)

# call local host
run_app(df, mode="external", host="127.0.0.1", port=5000)

# this should return the port http://127.0.0.1:5000/ that you must copy
# and paste in some browser (beware that this consumes CPU and local RAM)
