import warnings
import pandas as pd
from datetime import datetime
from hampel import hampel
from src.preprocessing.preprocessing import lower_rename, drop_nan_columns
from src.utils.visualizations import plot_time_series

warnings.filterwarnings("ignore")

# read data
path = "data/dataFlotacion.csv"
df = pd.read_csv(path)
print(df.isna().sum())
# there's a doble columns, so create a new one and deleting others
# ["Tph tratamiento"]
df["Tph tratamiento Original"] = df["Tph tratamiento"]

# delete non usable columns [client rule]
drop_columns = ['%Cu Cola SCV', '%Fe Cola SCV', '%Sol Cola SCV',
                '%Cu Cola final', "Tph tratamiento"]
df = df.drop(columns=drop_columns)
print(df.isna().sum())
# rename all columns to lower case (avoid future problems with databases)
df = lower_rename(df)

# datetime date
date_format = '%Y-%m-%d %H:%M:%S'
df["fecha"] = df["fecha"].apply(lambda x: datetime.strptime(x, date_format))

# droping cols with over 99 % of nans --> non usable
df = drop_nan_columns(df, threshold=99)

# target columns
target_cols = ['%cu_conc_final']

# set fecha as index
df.set_index(["fecha"], inplace=True)

# plot time series
plot_time_series(df, fecha_inicial="2018-01-01 00:00:00",
                 fecha_final="2020-03-10 05:30:00",
                 title="Evolution flotation variables",
                 ylabel="None",
                 sample=9)
df.reset_index(drop=False, inplace=True)

# order variables as actionable and non actionable and target
df = df[['fecha', 'tph_tratamiento_original', 'di-101',
         'espumante_std', 'xantato', 'nahs', 'recuperacion_global',
         '%cu_alimen._rougher', '%fe_alimen._rougher',
         '%sol_alimen._rougher', 'ph_rougher',
         '%cu_colas_rougher', '%fe_colas_rougher', '%sol_colas_rougher',
         '%cu_conc._rougher', '%cu_conc_limp_scv', '%fe_conc_limp_scv',
         '%_conc_limp_scv', '%as_conc_limp_scv',
         '%cu_concentrado_limpieza_rougher', '%feconcentrado_limpieza_rougher',
         '%sol_concentrado_limpieza_rougher',
         '%as_concentrado_limpieza_rougher',
         'ph_limpieza', 'ph_limpieza2', 'nivel_columna_500',
         'nivel_columna_501', 'nivel_columna_502', 'nivel_columna_503',
         'nivel_columna_504', 'nivel_columna_505', 'nivel_columna_506',
         'nivel_columna_507', 'nivel_columna_508', 'nivel_columna_509',
         'nivel_columna_510', 'nivel_columna_511', 'espumante_sag',
         'colector_primario', '%cu_conc_final']]

# resample on datetimes
df.set_index(["fecha"], inplace=True)

df_resample = df.resample('600S').mean()
print(df_resample.isna().sum())

interpolated = df_resample.interpolate(method='linear')
print(interpolated.isna().sum())

# outliers detection and imputation [using the mining shits
# window_size = (60 minutes/10 minutes) * 12 hours * 2 shifs* 7 days]
for col in interpolated.columns:
    df[col] = hampel(df[col], window_size=5, imputation=True)


plot_time_series(interpolated, fecha_inicial="2018-01-01 00:00:00",
                 fecha_final="2020-03-10 05:30:00",
                 title="Evolution flotation variables",
                 ylabel="None",
                 sample=9)

# reset_index and save
interpolated.reset_index(drop=False, inplace=True)
path_cleaned = "data/cleaned-data.csv"
interpolated.to_csv(path_cleaned, index=False,
                    date_format=date_format)
