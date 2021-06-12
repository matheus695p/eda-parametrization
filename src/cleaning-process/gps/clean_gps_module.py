import pytz
import boto3
import pickle
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime


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


def axis_time_movement(df):
    """
    Calcula el movimiento x,y e z en un delta T de movimiento
    Parameters
    ----------
    df : Dataframe
        gps data
    Returns
    -------
        gps data with movement on axis and time
    """
    df["diff_fecha"] = df.sort_values(['equipo', 'fecha']).\
        groupby('equipo')['fecha'].diff().dt.total_seconds()
    df["diff_este"] = df.sort_values(['equipo', 'fecha']).\
        groupby('equipo')['este'].diff()
    df["diff_norte"] = df.sort_values(['equipo', 'fecha']).\
        groupby('equipo')['norte'].diff()
    df["diff_cota"] = df.sort_values(['equipo', 'fecha']).\
        groupby('equipo')['cota'].diff()
    df["diff_vel"] = df.sort_values(['equipo', 'fecha']).\
        groupby('equipo')['velocidad'].diff()
    return df


def last_state_xyz(df):
    """
    Obtiene el último estado (x, y, z and speed) de cada evento de gps
    Parameters
    ----------
    df : dataframe
        gps data
    Returns
    -------
        4 columnas adicionales al estado
    """
    df['prev_este'] = df.sort_values(by=["equipo", "fecha"]).\
        groupby("equipo").shift(1)["este"]
    df['prev_norte'] = df.sort_values(by=["equipo", "fecha"]).\
        groupby("equipo").shift(1)["norte"]
    df['prev_cota'] = df.sort_values(by=["equipo", "fecha"]).\
        groupby("equipo").shift(1)["cota"]
    df['prev_vel'] = df.sort_values(by=["equipo", "fecha"]).\
        groupby("equipo").shift(1)["velocidad"]
    return df


def drop_spaces_data(df):
    """
    sacar los espacios de columnas que podrián venir interferidas

    Parameters
    ----------
    df : dataframe
        input data
    column : string
        string sin espacios en sus columnas
    Returns
    -------
    """
    for column in df.columns:
        try:
            df[column] = df[column].str.lstrip()
            df[column] = df[column].str.rstrip()
        except Exception as e:
            print(e)
            pass
    return df


def euclidian_distance(diff_x, diff_y, diff_z):
    """
    Determinar la distancia euclidiana entre el punto anterior y el actual
    de forma determinar la distancia en metros.
    Parameters
    ----------
    diff_x : float64
        diferencia en la coordenada x.
    diff_y : float64
        diferencia en la coordenada y.
    diff_z : float64
        diferencia en la coordenada z.
    Returns
    -------
    distance : float64
        movimiento discreto entre un punto x e x+1, donde x+1 es el registro
        siguiente.
    """
    distance = np.abs(
        diff_x*diff_x) + np.abs(diff_y*diff_y) + np.abs(diff_z*diff_z)
    distance = np.sqrt(distance)
    return distance


def resample_gps(gps, freq="5S"):
    """
    Hacer resample de los puntos de gps a freq segundos de proximidad
    Parameters
    ----------
    gps : dataframe
        dataframe de gps.
    freq : string, optional
        frecuencia de resampleo. The default is "5S".
    Returns
    -------
    salida : dataframe
        dataframe de gps resampleado.
    """
    salida = pd.DataFrame()
    # resample de gps
    for equipo in gps["equipo"].unique():
        print(equipo)
        gps_equipo = gps[gps["equipo"] == equipo]
        gps_equipo.sort_values(by=["fecha"], inplace=True)
        gps_equipo.reset_index(inplace=True, drop=True)
        gps_equipo.set_index(["fecha"], inplace=True)
        gps_equipo = gps_equipo.resample(freq).asfreq()
        # relleno de vacios
        gps_equipo["equipo"].fillna(value=equipo, inplace=True)
        gps_equipo["velocidad"].fillna(method="bfill", inplace=True)

        gps_equipo = gps_equipo.interpolate(
            method='linear', limit_direction='backward', axis=0)
        gps_equipo.reset_index(drop=False, inplace=True)
        salida = pd.concat([salida, gps_equipo], axis=0)
    salida.sort_values(by=["equipo", "fecha"], inplace=True)
    salida.reset_index(drop=True, inplace=True)
    return salida


def transform_date(date):
    """
    Transformar la fecha y formato de timestamp según el formato que venian
    de jigsaw
    Parameters
    ----------
    date : TYPE
        DESCRIPTION.

    Returns
    -------
    fecha_loads : TYPE
        DESCRIPTION.

    """
    fecha = date[0:10]
    hora = date[11:19]
    date = fecha + " " + hora
    fecha_loads = datetime.strptime(date,
                                    "%Y-%m-%d %H:%M:%S")
    fecha_loads = timezone_fechas("America/Santiago", fecha_loads)
    return fecha_loads


def read_pkl_s3(bucket, ruta):
    """
    La funcion lee un archivo pkl desde s3
    Parameters
    ----------
    bucket : Nombre del bucket
    ruta : Ruta del archivo
    Returns
    -------
    data : Dataframe con los datos
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket,
                        Key=ruta)
    body = obj["Body"].read()
    data = pickle.loads(body)
    data.reset_index(inplace=True, drop=True)
    return data


def read_csv_s3(s3_bucket, s3_event_key):
    """
    Leer archivo .csv desde s3

    Parameters
    ----------
    bucket : string
        nombre del bucket.
    ruta : string
        ruta de s3 ddonde ir a buscar la data.\
    Returns
    -------
    data : dataframe
        lectura del archivo como un pandas dataframe

    Alternativa:
    obj = s3.get_object(Bucket=s3_bucket,
                        Key=s3_event_key)
    body = io.BytesIO(obj["Body"].read())
    # leer la data
    data = pd.read_csv(body, sep=",",
                        index_col=0,
                        encoding="ISO-8859-1")

    """
    s3 = boto3.client("s3")
    csv_obj = s3.get_object(Bucket=s3_bucket, Key=s3_event_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(StringIO(csv_string), index_col=0)
    return data
