import boto3
import json
import pandas as pd
import pytz
from datetime import datetime
from storing_files import store_in_s3
from clean_mems_module import drop_spaces_data


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
    s3 = boto3.client("s3")
    # Definimos la fecha
    timezone = pytz.timezone("America/Santiago")
    fecha = datetime.now(timezone).replace(microsecond=0)
    # Traer el archivo
    obj = s3.get_object(Bucket=s3_bucket,
                        Key=s3_event_key)
    text = "["+obj["Body"].read().decode('utf-8').replace('\n', '').\
        replace('\ufeff', '') + "]"
    mems = pd.read_json(text)

    mems = pd.read_pickle(
        r"C:\Users\mateu\OneDrive\Desktop\proyectos\tires-maintenance\data\raw-data\mems\mems-2020-07-30.pkl")
    data_name = "mems"
    # solo en el caso de que hayan datos de mems
    if len(mems) > 0:
        # Procesamos la data
        columnas = ['Equipo', 'Date', 'Neumatico_Key', 'Presion', 'Temperatura',
                    'TipoVehiculo_Key', 'Vehiculo_Key', 'TipoVehiculo_Nombre']
        mems = mems[columnas]
        mems.rename(columns={"Equipo": "equipo",
                             "Vehiculo_Key": "key_equipo",
                             "TipoVehiculo_Nombre": "flota",
                             "Neumatico_Key": "key_neumatico",
                             "TipoVehiculo_Key": "key_tipo_vehiculo",
                             "Date": "date",
                             "Presion": "presion",
                             "Temperatura": "temperatura"}, inplace=True)
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
