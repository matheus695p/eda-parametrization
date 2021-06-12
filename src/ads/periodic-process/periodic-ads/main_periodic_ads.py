import os
import boto3
from time import time
from datetime import datetime, timedelta
from ads_module import (response_s3, timezone_fechas, gps_preprocessing,
                        loads_preprocessing, mems_preprocessing,
                        compute_ads, upload_files_as_pkl)


def handler(event, context):
    """
    Este código se encargara de hacer el proceso de feature engineering de los
    datos de gps, loads y mems

    Parameters
    ----------
    event : dict
        evento con el año-mes-dia hora:minuto:segundo del evento que se
        quiere calcular
        ejemplo: event = {'fecha': '2020-11-27 00:05:00'}.
    Returns
    -------
    ads : dataframe
        dataframe con el analytic dataset que fue construido en el proceso,
        a nivel de neumáticos
    """
    # inicio del conteo del tiempo
    start_time = time()
    # evento de prueba borrar después
    event = {"fecha": "2020-11-27 00:05:00"}
    try:
        fecha = event["fecha"]
        fecha = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
        fecha = timezone_fechas("America/Santiago", fecha)
        print("La fecha viene desde un evento: ", fecha)

    except Exception as e:
        print(e)
        print("Esta es una ejecución automática")
        fecha = datetime.now()
        fecha = timezone_fechas("America/Santiago", fecha)
        print("Esta es una ejecución automática: ", fecha)

    # dias de antigueadad de la data
    dias = 2
    # fecha de un dia atrás
    fecha_inicial = fecha - timedelta(days=dias)

    # cliente s3
    s3 = boto3.client("s3")
    # cliente ssm
    ssm = boto3.client("ssm")
    # prefijo de los caex para filtrar equipo
    prefijo = ssm.get_parameter(
        Name='PrefijoCaex', WithDecryption=True)["Parameter"]["Value"]
    # ir a buscar la data en s3
    gps = response_s3(s3, fecha, dias, "gps")
    loads = response_s3(s3, fecha, dias, "loads")
    mems = response_s3(s3, fecha, dias, "mems")
    # filtros por fechas y equipos
    gps = gps[(gps["date"] <= fecha) &
              (gps["date"] >= fecha_inicial) &
              (gps["equipo"].str.contains(prefijo))]
    gps.sort_values(by=["equipo", "date"], inplace=True)
    gps.reset_index(drop=True, inplace=True)
    loads = loads[(loads["date"] <= fecha) &
                  (loads["date"] >= fecha_inicial) &
                  (loads["equipo"].str.contains(prefijo))]
    loads.reset_index(drop=True, inplace=True)
    mems = mems[(mems["date"] <= fecha) &
                (mems["date"] >= fecha_inicial) &
                (mems["equipo"].str.contains(prefijo))]
    mems.reset_index(drop=True, inplace=True)
    # preprocesamiento de los datos para tener todas las columnas
    gps = gps_preprocessing(gps)
    loads = loads_preprocessing(loads)
    mems = mems_preprocessing(mems)

    # computo del ads a nivel de equipos
    ads = compute_ads(fecha, dias, gps, loads, mems, delta_hours=24)

    # enviar los datos a S3
    files = [["ads", ads, "ads",
              os.environ["outputbucket"]]]
    upload_files_as_pkl(files, fecha)
    elapsed_time = time() - start_time
    print("Tiempo transcurrido: %0.10f seconds." % elapsed_time)
