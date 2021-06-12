import os
import pytz
import boto3
import warnings
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")


def ls(path):
    lista = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if ".pkl" in filename:
                print(filename)
                fecha = filename[4:14]
                lista.append(fecha)
    return lista


def main(fecha_final):
    # subir los datos de gps
    send_fuente(fecha_final, fuente="gps")
    # subir los datos de mems
    send_fuente(fecha_final, fuente="mems")
    # subir los datos de loads
    send_fuente(fecha_final, fuente="loads")
    # subir los datos de maintenance
    # send_fuente(fecha_final, fuente="maintenance")


def send_fuente(fecha_final, fuente="mems", delta_days=20):
    s3 = boto3.client("s3")
    bucket_name = "tiresopt-anglo-raw"
    path =\
        fr'C:/Users/mateu/OneDrive/Desktop/proyectos/tires-optimizer/data/raw-data/{fuente}/{fuente}-{fecha_final}.pkl'
    print(path)
    data = pd.read_pickle(path)
    # filtro de fechas
    fecha = datetime.strptime(
        fecha_final, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    fecha_final = timezone_fechas("America/Santiago", fecha)
    fecha_inicial = fecha_final - timedelta(days=delta_days)
    fecha_inicial = fecha_inicial.replace(hour=0, minute=0, second=0)

    # filtro en las fechas indicadas
    data = data[(data["Date"] >= fecha_inicial) &
                (data["Date"] <= fecha_final)]
    data.sort_values(by=["Date"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    head = data.head(5000)
    print(head)
    data["day"] = data["Date"].dt.day
    data["month"] = data["Date"].dt.month
    data["day"] = data["day"].apply(str)
    data["month"] = data["month"].apply(str)
    data["fecha"] = data["day"] + "-" + data["month"]
    data.drop(columns=["day", "month"], inplace=True)

    for col in data.columns:
        type_ = str(data[col].dtype)
        print("nada", col, type_)
        if "datetime" in type_:
            print(col, type_)
            data[col] = data[col].apply(str)

    fechas = data["fecha"].unique()
    for fecha in fechas:
        print("Subiendo fecha: ", fecha)
        data_red = data[data["fecha"] == fecha]
        data_red.drop(columns=["fecha"], inplace=True)
        # formato del archivo
        format_ = ".csv"
        key = f"{fuente}/{fuente}-{fecha}{format_}"
        print(key)

        data_red.to_csv(f"{fuente}{format_}", sep=",")
        # data.to_json(f"{fuente}{format_}", orient="records")
        s3.upload_file(Filename=f"{fuente}{format_}",
                       Bucket=bucket_name,
                       Key=key)
        os.remove(f"{fuente}{format_}")


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


def ls_folder(path):
    for file in os.listdir(path):
        print(file)


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def upload_from_local(bucket_name, local_path, prefix):
    s3 = boto3.client("s3")
    for r, d, f in os.walk(local_path):
        for file in f:
            if file.endswith(".json"):
                alpha = os.path.join(r, file)
                filename = alpha.replace(local_path, "")
                # key
                key = prefix + filename
                print(alpha)
                print(key)
                s3.upload_file(Filename=alpha,
                               Bucket=bucket_name,
                               Key=key)


path = fr'C:/Users/mateu/OneDrive/Desktop/proyectos/tires-optimizer/data/raw-data/gps'
lista_fechas = ls(path)

lista_fechas = [
    '2020-07-15',
    '2020-07-30',
    '2020-08-14',
    '2020-08-29',
    '2020-09-13',
    '2020-09-28',
    '2020-10-13',
    '2020-10-28',
    '2020-11-12',
    '2020-11-27']

for fecha in lista_fechas:
    print(fecha)
    main(fecha)

# if __name__ == '__main__':
#     main()
