import pytz
import json
import boto3
from datetime import datetime
from storing_files import store_in_s3
from clean_gps_module import (drop_spaces_data, transform_date, read_csv_s3)


def handler(event, context):
    """
    Limpia los datos de gps, escuchando un tópico de SNS que avisa la creación
    de un nuevo objeto
    """
    print(event)
    s3_event_key = (json.loads(event['Records'][0]['Sns']['Message'])
                    ["Records"][0]["s3"]["object"]["key"])
    s3_bucket = (json.loads(event['Records'][0]['Sns']['Message'])
                 ["Records"][0]["s3"]["bucket"]["name"])
    print(s3_event_key)
    print(s3_bucket)
    # llamar a s3
    s3 = boto3.client("s3")
    print("Imprimir los buckets disponibles", s3.list_buckets())
    # Definimos la fecha actual para grabarla como fecha de llegada del archivo
    timezone = pytz.timezone("America/Santiago")
    fecha = datetime.now(timezone).replace(microsecond=0)
    # Traer el objecto en la ruta especificada
    gps = read_csv_s3(s3_bucket, s3_event_key)
    # nombre con el que serán guardados los datos
    data_name = "gps"
    # solo cuando gps es mayor a cero se hace este proceso
    if len(gps) > 0:
        print("Entro al procesamiento de la data")
        # Procesamos la data
        columnas = ['Equipo', 'Date', 'Velocidad', 'Norte', 'Este', 'Cota']
        gps = gps[columnas]
        gps.rename(columns={"Equipo": "equipo", "Date": "date",
                            "Velocidad": "velocidad", "Este": "este",
                            "Norte": "norte", "Cota": "cota"},
                   inplace=True)
        # eliminar los espacios en las columnas
        gps = drop_spaces_data(gps)
        # transformar a datetime
        gps["date"] = gps["date"].apply(lambda x: transform_date(x))

        gps.drop_duplicates(subset=["equipo", "date"], inplace=True)
        # eliminar los que no traen datos de equipos
        gps.dropna(subset=["equipo"], inplace=True)
        gps["date_gps"] = gps["date"]
        gps["date_arrive"] = fecha
        # reset index
        gps.reset_index(drop=True, inplace=True)
        # Guardamos la data en el formato correcto en s3
        store_in_s3(s3, data_name, gps, drop=["equipo", "date"],
                    columna="date")
        print(f"Terminé de procesar {data_name}")
    else:
        print(f"No hay data que guardar en {data_name}")
