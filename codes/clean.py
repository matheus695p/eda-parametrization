import warnings
import pandas as pd
from src.preprocessing import lower_rename, drop_nan_columns
from src.visualizations import plot_time_series

warnings.filterwarnings("ignore")

# lectura y transformación de los datos
df = pd.read_csv("data/dataFlotacion.csv")
print(df.isna().sum())

# columans a borrar
drop_columns = ['%Cu Cola SCV', '%Fe Cola SCV', '%Sol Cola SCV']

# borrar columnas que no sirven
df = df.drop(columns=drop_columns)

# convertir nombres de las columnas
df = lower_rename(df)

# borrar columnas con un indice de nans mayor al cero porciento
df = drop_nan_columns(df, index=0)

# columnas target
target_cols = ['%cu_conc_final', '%cu_cola_final']

# setiar fecha como indice
df.set_index(["fecha"], inplace=True)

# plot de la serie de tiempo
plot_time_series(df, fecha_inicial="2018-01-01 00:00:00",
                 fecha_final="2020-03-10 05:30:00",
                 title="Evolución de variables flotación",
                 ylabel="None",
                 sample=9)
# resetear indice
df.reset_index(drop=False, inplace=True)

# ordenar las variables por variables accionables / no accionables / targets

df = df[['fecha', 'tph_tratamiento', 'tph_tratamiento', 'di-101',
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
         'colector_primario', '%cu_conc_final', '%cu_cola_final']]

df.to_csv("data/cleaned-data.csv", index=False)
