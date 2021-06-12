import pytz
import json
import boto3
from datetime import datetime
from storing_files import store_in_s3
from clean_loads_module import (drop_spaces_data, read_csv_s3,
                                transform_date)


def handler(event, context):
    """
    Limpia los datos que contienen las cargas y descargas de mineral, la
    lambda esta escuchando un tópico de SNS el cual gatilla la ejecución
    El resultado se guarda en el bucket de entrada
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
    loads = read_csv_s3(s3_bucket, s3_event_key)
    print(loads.head())
    # nombre con el que será guardado en el github
    data_name = "loads"

    if len(loads) > 0:
        # Procesamos la data
        columnas = ['IdTurno', 'Equipo', 'Date', 'Origen', 'UbicacionDescarga',
                    'Date descarga inicio', 'Date descarga fin', 'Toneladas',
                    'Operador', 'Grupo']
        loads = loads[columnas]
        loads.rename(columns={"IdTurno": "id_turno", "Equipo": "equipo",
                              "Date": "date", "Origen": "origen",
                              "UbicacionDescarga": "destino",
                              "Date descarga inicio": "fecha_inicio_descarga",
                              "Date descarga fin": "fecha_fin_descarga",
                              "Toneladas": "tonelaje",
                              "Operador": "operador",
                              "Grupo": "grupo"},
                     inplace=True)
        # arreglar las fechas

        # agregar los nombres de los operadores
        loads["operador"] = loads["operador"].str.lower().str.strip()
        loads["operador"] = loads["operador"].fillna("nr")
        # reemplazar grupos nombre
        loads["grupo"] = loads["grupo"].str.replace(
            "Grupo ", "G").replace(" ", "")
        # eliminar aquellos que no han descargado aún
        loads.dropna(subset=["fecha_inicio_descarga"], inplace=True)
        loads.dropna(subset=["fecha_fin_descarga"], inplace=True)
        # eliminar espacios iniciales y finales
        loads = drop_spaces_data(loads)

        # transformar las fechas
        loads["date"] = loads["date"].apply(lambda x: transform_date(x))
        # transformar fecha_inicio_descarga
        loads["fecha_inicio_descarga"] =\
            loads["fecha_inicio_descarga"].apply(lambda x: transform_date(x))
        # transformar fecha_fin_descarga
        loads["fecha_fin_descarga"] =\
            loads["fecha_fin_descarga"].apply(lambda x: transform_date(x))

        # eliminar los duplicados
        loads.drop_duplicates(subset=["equipo", "date"], inplace=True)
        # borrar los que no tienen equipo
        loads.dropna(subset=["equipo"], inplace=True)
        # ver la fecha de llegada de los datos
        loads["date_arrive"] = fecha
        # loads fecha
        loads["date_loads"] = loads["date"]
        loads.reset_index(drop=True, inplace=True)
        print(loads.shape)
        # Guardamos la data en el formato correcto en s3
        store_in_s3(s3, data_name, loads, drop=["equipo", "date"],
                    columna="date")
        # Obtenemos la lista de los operadores
        # operadores = list(loads["operador"].unique())

        # """
        # En esta parte se va invocar la función de registro de operadores,
        # las cuales va a hacer un registro de los operadores con un id único
        # """
        # # Enviamos operadores al registro
        # funcion = os.environ["registrooperador"]
        # client = boto3.client('lambda')
        # parametros = json.dumps(operadores)
        # response = client.invoke(FunctionName=funcion,
        #                          Payload=parametros,
        #                          InvocationType='Event')
        # print("Se acaba de invocar una lambda", response)
        print(f"Terminé de procesar {data_name}")

    else:
        print(f"No hay data que guardar en {data_name}")
