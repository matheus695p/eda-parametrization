import os
import pytz
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")


def ls(path):
    lista = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if ".pkl" in filename:
                print(filename)
                lista.append(filename)
    return lista


def rename_columns(df, prefix="mean"):
    """
    Renombra los nombres de las columnas de un dataframe con un prefijo
    Parameters
    ----------
    df : dataframe
        dataframe cualquiera.
    prefix : string, optional
        prefijo que se le quiere agregar al nombre de la columna.
        The default is "mean".
    Returns
    -------
    df : dataframe
        dataframe con las columnas renombradas.
    """
    df.reset_index(drop=True, inplace=True)
    for col in df.columns:
        if "equipo" in col:
            pass
        elif "key_neumatico" in col:
            pass
        else:
            new_name = prefix + "_" + col
            df.rename(columns={col: new_name}, inplace=True)
    return df


def timezone_fechas(zona, fecha):
    """
    La funcion entrega el formato de zona horaria a las fechas de los
    dataframe
    Parameters
    ----------
    zona: zona horaria a usar
    fecha: fecha a modificar
    Returns
    -------
    fecha_zh: fecha con el fomarto de zona horaria
    """
    # Definimos la zona horaria
    timezone = pytz.timezone(zona)
    fecha_zs = timezone.localize(fecha)
    return fecha_zs


path =\
    fr'C:/Users/mateu/OneDrive/Desktop/proyectos/tires-optimizer/data/raw-data/mems'
path_salida =\
    r'C:/Users/mateu/OneDrive/Desktop/proyectos/tires-optimizer/data/input-data/mems/maintenance'
archivos = ls(path)
fecha_futuro = datetime.strptime("2100-01-01", "%Y-%m-%d")
fecha_futuro = timezone_fechas("America/Santiago", fecha_futuro)
fecha_pasado = datetime.strptime("2000-01-01", "%Y-%m-%d")
fecha_pasado = timezone_fechas("America/Santiago", fecha_pasado)

columnas = ['Equipo', 'Date', 'Neumatico_Key', 'Presion',
            'Temperatura', 'TipoVehiculo_Key', 'Vehiculo_Key',
            'TipoVehiculo_Nombre']
contador = 0
for archivo in archivos:
    new_path = path + "/" + archivo
    print(new_path)
    data = pd.read_pickle(new_path)
    data = data[columnas]
    data.rename(columns={"Equipo": "equipo",
                         "Vehiculo_Key": "key_equipo",
                         "TipoVehiculo_Nombre": "flota",
                         "Neumatico_Key": "key_neumatico",
                         "TipoVehiculo_Key": "key_tipo_vehiculo",
                         "Date": "date",
                         "Presion": "presion",
                         "Temperatura": "temperatura"}, inplace=True)
    data = data[data["flota"] == "K930"]
    data = data[["equipo", "key_neumatico", "date"]]
    data.reset_index(drop=True, inplace=True)
    max_date = data.groupby(["equipo", "key_neumatico"])[["date"]].max()
    max_date.columns = ["fecha_maxima"]
    max_date.reset_index(drop=False, inplace=True)

    min_date = data.groupby(["equipo", "key_neumatico"])[["date"]].min()
    min_date.columns = ["fecha_minima"]
    min_date.reset_index(drop=False, inplace=True)

    fechas = min_date.merge(max_date, on=["equipo", "key_neumatico"],
                            how="outer")
    name = path_salida + "/" + "maintenance_data.pkl"
    if contador == 0:
        fechas.to_pickle(name)
    contador += 1
    # nuevo dataframe de fechas
    nuevo = rename_columns(fechas, prefix="nuevo")
    # viejo dataframe de fechas
    antiguo = pd.read_pickle(name)
    antiguo = rename_columns(antiguo, prefix="antiguo")
    # combinación de ambos dataframe
    comb = nuevo.merge(antiguo, on=["equipo", "key_neumatico"], how="outer")
    comb = comb.replace({pd.NaT: np.nan})

    for col in comb.columns:
        if "maxima" in col:
            comb[col].fillna(value=fecha_pasado, inplace=True)
        if "minima" in col:
            comb[col].fillna(value=fecha_futuro, inplace=True)

    comb["fecha_minima"] = comb[
        ["antiguo_fecha_minima", "nuevo_fecha_minima"]].min(axis=1)
    comb["fecha_maxima"] = comb[
        ["antiguo_fecha_maxima", "nuevo_fecha_maxima"]].max(axis=1)
    comb = comb[["equipo", "key_neumatico", "fecha_minima", "fecha_maxima"]]
    comb.to_pickle(name)


fecha_inicial = "2020-07-08"
fecha_inicial = datetime.strptime(fecha_inicial, "%Y-%m-%d")
fecha_inicial = timezone_fechas("America/Santiago", fecha_inicial)

fecha_final = "2021-02-22"
fecha_final = datetime.strptime(fecha_final, "%Y-%m-%d")
fecha_final = timezone_fechas("America/Santiago", fecha_final)


comb["diff_date"] = (comb["fecha_maxima"] - comb[
    "fecha_minima"]).dt.total_seconds() / (3600 * 24)
comb.sort_values(by=["equipo", "fecha_minima"], inplace=True)

validos = comb[(comb["fecha_minima"] > fecha_inicial) &
               (comb["fecha_maxima"] < fecha_final)]
