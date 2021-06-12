import pytz
import boto3
import pickle
import pandas as pd
from io import StringIO
from datetime import datetime


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
