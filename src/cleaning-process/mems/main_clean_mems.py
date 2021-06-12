import pytz
import json
import boto3
from datetime import datetime
from storing_files import store_in_s3
from clean_mems_module import (drop_spaces_data, transform_date, read_csv_s3)


def handler(event, context):
    """
    Limpia y lleva al formato correcto los datos de temperatura y
    presion de neumaticos
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
    # Definimos la fecha
    timezone = pytz.timezone("America/Santiago")
    fecha = datetime.now(timezone).replace(microsecond=0)
    # Traer el archivo
    mems = read_csv_s3(s3_bucket, s3_event_key)
    data_name = "mems"
    # solo en el caso de que hayan datos de mems
    if len(mems) > 0:
        # Procesamos la data
        columnas = ['Equipo', 'Date', 'Neumatico_Key', 'Presion',
                    'Temperatura', 'TipoVehiculo_Key', 'Vehiculo_Key',
                    'TipoVehiculo_Nombre']
        mems = mems[columnas]
        mems.rename(columns={"Equipo": "equipo",
                             "Vehiculo_Key": "key_equipo",
                             "TipoVehiculo_Nombre": "flota",
                             "Neumatico_Key": "key_neumatico",
                             "TipoVehiculo_Key": "key_tipo_vehiculo",
                             "Date": "date",
                             "Presion": "presion",
                             "Temperatura": "temperatura"}, inplace=True)
        # transformar date
        mems["date"] = mems["date"].apply(lambda x: transform_date(x))
        # sacar los espacios al inicio y final de cada columna
        mems = drop_spaces_data(mems)
        # transformar date
        # mems["date"] = mems["date"].apply(lambda x: transform_date(x))
        # agregar fecha de llegada del dato
        mems["date_arrive"] = fecha
        # fecha de mems
        mems["date_mems"] = mems["date"]
        mems.drop_duplicates(subset=["equipo", "date", "key_neumatico"],
                             inplace=True)
        # eliminar los que no tengan equipo
        mems.dropna(subset=["equipo"], inplace=True)
        mems.reset_index(drop=True, inplace=True)
        # Guardamos la data en el formato correcto en s3
        store_in_s3(s3, data_name, mems, drop=["equipo", "date",
                                               "key_neumatico"],
                    columna="date")
        print(f"Terminé de procesar {data_name}")
    else:
        print(f"No hay data que guardar en {data_name}")
