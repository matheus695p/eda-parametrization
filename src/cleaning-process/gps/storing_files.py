import io
import os
import pytz
import boto3
import pandas as pd
from datetime import datetime, timedelta


def store_in_s3(s3, data_name, df, drop=["date", "equipo"], columna="date"):
    """
    La funcion guarda la data en archivos seprados por dia en función de lo
    que encuentra en un mismo día
    Parameters
    ----------
    s3 : Cliente s3
    data_name : De que es la data a analizar
    df : DataFrame de la data a guardar
    """
    # Identificamos todos los distintos dias que hay en el archivo
    unique_days = df[columna].map(lambda t: t.date()).unique()
    bucket = os.environ["inputbucket"]
    # iteramos sobre los días que hay disponibles
    for dia in unique_days:
        print(dia)
        print(data_name)
        day_time = datetime.combine(dia, datetime.min.time())
        next_day = dia + timedelta(days=1)
        next_day_time = datetime.combine(next_day, datetime.min.time())
        try:
            data_to_store = df[(df[columna] >= day_time) &
                               (df[columna] < next_day_time)]
        except Exception as e:
            print(e)
            day_time = timezone_fechas("America/Santiago", day_time)
            next_day_time = timezone_fechas("America/Santiago", next_day_time)
            data_to_store = df[(df[columna] >= day_time) &
                               (df[columna] < next_day_time)]
        # Definimos la ruta a guardar
        year_number = day_time.year
        month_number = day_time.month
        day_number = day_time.day
        rute_store = (f'{data_name}-preprocessed/{year_number}/{month_number}/'
                      f'{day_number}/{data_name}')
        if len(data_to_store) > 0:
            try:
                buffer = io.BytesIO()
                client = boto3.resource("s3")
                object = client.Object(bucket,
                                       f"{rute_store}.parquet")
                object.download_fileobj(buffer)
                df2 = pd.read_parquet(buffer)
                print(f"archivo {rute_store} cargado")
            except Exception as e:
                print(e)
                print(f"El archivo {rute_store} aun no existe")
                df2 = pd.DataFrame()

            data_to_store = pd.concat([df2, data_to_store])
            data_to_store.drop_duplicates(subset=drop, inplace=True)

            # guarda en formato pickle
            # data_to_store.to_pickle("/tmp/data.pkl")
            # s3.upload_file(Filename="/tmp/data.pkl",
            #                Bucket=bucket,
            #                Key=f"{rute_store}.pkl")

            data_to_store.to_parquet("/tmp/data.parquet")
            s3.upload_file(Filename="/tmp/data.parquet",
                           Bucket=bucket,
                           Key=f"{rute_store}.parquet")
            print(data_to_store[columna].head(1))

            try:
                buffer = io.BytesIO()
                client = boto3.resource("s3")
                object = client.Object(bucket,
                                       f"{rute_store}.parquet")
                object.download_fileobj(buffer)

                df3 = pd.read_parquet(buffer)
                print(df3[columna].head(1))
            except Exception as e:
                print(e)
                print("#odioparquet", "fallo de parquet")
        else:
            print("No hay data que guardar")


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


def last_date(s3, df, fuente, columna):
    """
    La funcion envia la ultima fecha en el archivo a S3
    Parameters
    ----------
    s3 : Cliente s3
    df : DataFrame al que se le buscara la ultima fecha
    fuente : Nombre de la fuente de datos
    """
    df.sort_values(by=[columna], inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)
    last_date = df.loc[0, columna]
    archivo = pd.DataFrame([last_date])
    nombre_archivo = fuente + ".pkl"
    archivo.to_pickle("/tmp/" + nombre_archivo)
    bucket = os.environ["inputbucket"]
    s3.upload_file(Filename="/tmp/" + nombre_archivo,
                   Bucket=bucket,
                   Key='Chequeo_datos/' + nombre_archivo)
