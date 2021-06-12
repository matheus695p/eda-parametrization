import io
import os
import pytz
import boto3
import warnings
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from datetime import timedelta
warnings.filterwarnings("ignore")


def response_s3(s3, fecha_actual, dias, ruta_s3):
    """
    La funcion busca los archivos .parquet que cumplen un rango de
    fechas
    Parameters
    ----------
    s3: Conexion a servicio s3
    fecha_actual : Ultima fecha desde la que se analiza
    dias : Dias hacia atras que se quiere buscar
    ruta_s3 : carpeta s3 a la que se quiere ir
    Returns
    -------
    data_final : dataframe con los datos
    """
    # Data final
    data_final = pd.DataFrame()
    # Lista de fechas en el rango
    fechas = []
    bucket = os.environ["inputbucket"]
    # Calculamos los dias anteriores
    for i in range(dias):
        d = (fecha_actual-timedelta(i)).day
        m = (fecha_actual-timedelta(i)).month
        y = (fecha_actual-timedelta(i)).year
        prefijo = str(y) + "/" + str(m) + "/" + str(d) + "/"
        fechas.append(prefijo)
    # Para cada dia
    for fecha in fechas:
        try:
            buffer = io.BytesIO()
            client = boto3.resource("s3")
            object = client.\
                Object(bucket,
                       f"{ruta_s3}-preprocessed/{fecha}{ruta_s3}.parquet")
            print(f"{ruta_s3}-preprocessed/{fecha}{ruta_s3}.parquet")
            object.download_fileobj(buffer)
            data = pd.read_parquet(buffer)
            data_final = pd.concat([data_final, data])
        except Exception as e:
            print(e)
            print("Falló en la ruta",
                  f"{ruta_s3}-preprocessed/{fecha}{ruta_s3}.parquet")
            pass
    return data_final


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


def round_date(fecha):
    """
    Redondear la fecha a los 30 segundos más cercanos
    Parameters
    ----------
    fecha : TYPE
        DESCRIPTION.

    Returns
    -------
    fecha_salida : datetime
        fecha redondeada a los 30 segundos más cercanos.
    """
    try:
        fecha_salida = fecha.round("30s")
    except Exception as e:
        print(e, "Fecha imposible de redondear")
        fecha_salida = fecha
    return fecha_salida


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
    df["diff_date"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['date'].diff().dt.total_seconds()
    df["diff_norte"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['norte'].diff()
    df["diff_este"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['este'].diff()
    df["diff_cota"] = df.sort_values(['equipo', 'date']).\
        groupby('equipo')['cota'].diff()
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
    df['prev_este'] = df.sort_values(by=["equipo", "date"]).\
        groupby("equipo").shift(1)["este"]
    df['prev_norte'] = df.sort_values(by=["equipo", "date"]).\
        groupby("equipo").shift(1)["norte"]
    df['prev_cota'] = df.sort_values(by=["equipo", "date"]).\
        groupby("equipo").shift(1)["cota"]
    return df


def slope_state(angulo, lim=4):
    """
    En función de la pendiente que va avanzando el camión, se define un estado
    de inclinación a definir en función de la distribución de cada mina

    Parameters
    ----------
    angulo : int
        angulo en grados.
    lim : int, optional
        limite de angulo para ser considerado en subida. The default is 4.
    Returns
    -------
    estado : TYPE
        retorna el estado en función del grado de inclinación, plano, subiendo
        o bajando.
    """
    if angulo > lim:
        estado = "subiendo"
    elif angulo < -lim:
        estado = "bajando"
    else:
        estado = "plano"
    return estado


def gps_preprocessing(df):
    """
    Hacer preprocesamiento con los datos de gps, encontrando las estimaciones
    de velocidad, angulo de inclinación con estado de inclinación

    Parameters
    ----------
    df : dataframe
        gps data
    Returns
    -------
        gps-preprocessed.
        unidades de gps
        aceleración: [m/s2]
        angulo: [grados]
        velocidad: [km/hr]
    """
    # cliente ssm
    ssm = boto3.client("ssm")
    # prefijo de los caex
    velocidad_maxima = 1.1 * float(ssm.get_parameter(
        Name='VelocidadMaxima', WithDecryption=True)["Parameter"]["Value"])

    if len(df) > 0:
        # sacar últimos estados
        df = axis_time_movement(df)
        df = last_state_xyz(df)
        # calcular movimiento en metros
        df["movimiento"] = df.apply(
            lambda x: euclidian_distance(
                x["diff_este"], x["diff_norte"], x["diff_cota"]), axis=1)
        # calcular velocidad estimada con gps
        df["velocidad_estimada"] = df["movimiento"] / df["diff_date"] * 3.6
        # filtrar aquellos valores de velocidad por sobre vel_max
        df["velocidad_estimada"] = df["velocidad_estimada"].apply(
            lambda x: velocidad_maxima if x > velocidad_maxima else x)
        # cálculo de la aceleración
        df["diff_vel_estimada"] = df.sort_values(['equipo', 'date']).\
            groupby('equipo')['velocidad_estimada'].diff()
        # aceleración en metros por segundo
        df["aceleracion"] = (df["diff_vel_estimada"] / 3.6) / df["diff_date"]
        # aceleración corregida a solo un valor positivo
        df["aceleracion_positiva"] = df["aceleracion"].apply(
            lambda x: 0 if x < 0 else x)
        df["aceleracion_negativa"] = df["aceleracion"].apply(
            lambda x: 0 if x > 0 else x)
        # calcular angulo de inclinación
        df["angulo"] = np.arcsin(
            df["diff_cota"] / df["movimiento"]) * 180 / np.pi
        # angulo en radianes
        df["angulo_rad"] = np.arcsin(
            df["diff_cota"] / df["movimiento"])
        # rellenar valores con cero, aquellos donde el movimiento fue cero
        df["angulo"].fillna(value=0, inplace=True)
        df["angulo_rad"].fillna(value=0, inplace=True)
        # angulo positivo y negativo
        df["angulo_positivo"] = df["angulo"].apply(lambda x: 0 if x < 0 else x)
        df["angulo_negativo"] = df["angulo"].apply(lambda x: 0 if x > 0 else x)
        # etiquetar el estado de la pendiente
        df["estado_pendiente"] = df["angulo"].apply(lambda x: slope_state(x))
        # sacar dataframe de salida
        df.sort_values(by=["equipo", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        pass
    return df


def loads_preprocessing(df):
    """
    Hacer el preprocesamiento de las cargas y descargas de mineral
    Parameters
    ----------
    df : dataframe
        dataframe de cargas y descargas de mineral.
    Returns
    -------
    df : dataframe
        dataframe con cantidades setiadas como reales
    """
    if len(df) > 0:
        # sacar las cargas en donde la fecha de fin de carga sea mayor
        df = df[df["fecha_fin_descarga"] > df["fecha_inicio_descarga"]]
        df.reset_index(inplace=True, drop=True)
        # setiar limites máximos y mínimos de en las toneladas
        df["tonelaje"] = df["tonelaje"].\
            apply(lambda x: 300 if x > 400 else x)
        df["tonelaje"] = df["tonelaje"].\
            apply(lambda x: 300 if x < 100 else x)
        # sacar el tiempo promedio entre carga y carga de mineral
        df['diff_cargas_min'] = df.sort_values(by=['equipo', 'date']).\
            groupby('equipo')['date'].diff().dt.total_seconds() / 60
        t_cargas_95 = df['diff_cargas_min'].quantile(0.95)
        t_cargas_median = df['diff_cargas_min'].median()
        # control de extremos y vacios
        df['diff_cargas_min'] = df['diff_cargas_min'].\
            apply(lambda x: t_cargas_median if x > t_cargas_95 else x)
        df['diff_cargas_min'].fillna(value=t_cargas_median, inplace=True)
        # ordenar la data
        df.sort_values(by=["equipo", "fecha_fin_descarga"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        pass
    return df


def mems_preprocessing_full(df):
    """
    Pre-procesamiento de mems con un undersampling de los datos

    Parameters
    ----------
    df : dataframe
        data de mems con presión y temperatura de los neumáticos.
    Returns
    -------
    salida : dataframe
        data de mems con presión y temperatura de los neumáticos preprocesada.
    """
    df = df[['equipo', 'date', 'key_neumatico', 'presion',
             'temperatura']]
    df.sort_values(by=["equipo", "date", "key_neumatico"], inplace=True)
    # daframe que traera la salida
    salida = pd.DataFrame()
    # stats por equipo
    for equipo in df["equipo"].unique():
        df_i = df[df["equipo"] == equipo]
        df_i["equipo"] = equipo
        # stats por neumatico
        for neumatico in df_i.key_neumatico.unique():
            df_j = df_i[df_i["key_neumatico"] ==
                        neumatico].reset_index(drop=True)
            df_j.sort_values(by=['date'], inplace=True)
            # Resampling de la data
            df_j.set_index("date", inplace=True)
            df_j = df_j.resample("60s").mean().reset_index(drop=False)
            df_j.fillna(method='ffill', inplace=True)
            df_j["key_neumatico"] = neumatico
            df_j["equipo"] = equipo
            df_j.reset_index(drop=True, inplace=True)
            salida = pd.concat([salida, df_j], axis=0)
    salida["date_round"] = salida["date"].apply(lambda x: round_date(x))
    salida = salida.groupby(by=["equipo", "date"]).mean()
    salida.reset_index(drop=False, inplace=True)
    # Sacar las diferencias de presión, hora y temperatura
    salida.reset_index(drop=True, inplace=True)
    return salida


def mems_preprocessing(df):
    """
    Pre-procesamiento de mems

    Parameters
    ----------
    df : dataframe
        data de mems con presión y temperatura de los neumáticos.
    Returns
    -------
    salida : dataframe
        data de mems con presión y temperatura de los neumáticos preprocesada.
    """
    if len(df) > 0:
        df = df[['equipo', 'date', 'key_neumatico', 'presion',
                 'temperatura']]
        df.sort_values(by=["equipo", "key_neumatico", "date"], inplace=True)
        # sacar las diferencias de temperatura
        df["diff_temp"] = df.sort_values(['equipo', 'key_neumatico', 'date']).\
            groupby(['equipo', 'key_neumatico'])['temperatura'].diff()
        # aumentos de temperatura
        df["diff_temp_postiva"] = df["diff_temp"].apply(
            lambda x: 0 if x < 0 else x)
        # bajas de temperatura
        df["diff_temp_negativa"] = df["diff_temp"].apply(
            lambda x: 0 if x > 0 else x)
        # diferencia de presiones en el tiempo
        df["diff_presion"] = df.sort_values(
            ['equipo', 'key_neumatico', 'date']).groupby(
                ['equipo', 'key_neumatico'])['presion'].diff()
        # aumentos de presión
        df["diff_presion_postiva"] = df["diff_presion"].apply(
            lambda x: 0 if x < 0 else x)
        # bajas de presión
        df["diff_presion_negativa"] = df["diff_presion"].apply(
            lambda x: 0 if x > 0 else x)
        df.reset_index(drop=True, inplace=True)
    else:
        pass
    return df

# def plot_mems(df, column="presion"):
#     if column == "presion":
#         unidad = "[bar]"
#     else:
#         unidad = "[°C]"
#     plt.rcParams.update({'font.size': 20})
#     fig, ax = plt.subplots(figsize=(25, 16))
#     for key in df["key_neumatico"].unique():
#         df_j = df[df["key_neumatico"] == key]
#         equipo = df_j["equipo"].unique()[0]
#         df_j.reset_index(drop=True, inplace=True)
    # ax.plot(df_j["date"], df_j[column],
    #         label=f'{column} / {equipo}-{key}')
#     ax.set_xlabel('Tiempo [freq=10-30s]')
#     ax.set_ylabel(f'{column} {unidad}')
#     ax.set_title(f'{column} {unidad}: {equipo}', fontsize=25)
#     ax.grid(True)
#     ax.legend(loc='upper left')
#     filename = f"imagenes/{equipo}-{column}.png"
#     fig.savefig(filename, dpi=600)


def round_1min(fecha):
    """
    Redondear fecha al minuto más cercano\
    Parameters
    ----------
    fecha : datetime
        fecha que se quiere redondear al minuto más cercano.
    Returns
    -------
    salida : datetime
        fecha redondeada.

    """
    try:
        salida = fecha.round("1min")
    except Exception as e:
        print(e)
        salida = 0
    return salida


def get_dates_loads(df, lista):
    """
    En las fechas de loads va apendeando el equipo con 1 entre los momentos
    que se estuvo cargando mineral
    Parameters
    ----------
    df : dataframe
        cargas y descargas de mineral.
    lista : list
        lista vacia.
    Returns
    -------
        Momnentos en los que un camión estuvo cargando o descargando mineral.
    """
    for i in pd.date_range(start=df["date"], end=df["fecha_fin_descarga"],
                           freq="min").values:
        lista.append([i, df["equipo"], 1])


def filter_by_day(df, fecha):
    """
    Filtrar solo los datos de la fecha en la que se encuentra solamente

    Parameters
    ----------
    df : dataframe
        alguna de las fuentes de datos.
    fecha : datetime
        fecha actual del ads.
    Returns
    -------
    df : dataframe
        fuente de datos filtrada por el último dia.
    """
    day = fecha.day
    df["day"] = df["date"].dt.day
    df = df[df["day"] == day]
    df.drop(columns=["day"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def filter_by_hours(df, fecha, delta_hours):
    """
    Filtrar solo los datos de la fecha en la que se encuentra solamente

    Parameters
    ----------
    df : dataframe
        alguna de las fuentes de datos, que contenga la columna ["date"].
    fecha : datetime
        fecha actual del ads.
    Returns
    -------
    df : dataframe
        fuente de datos filtrada por el último dia.
    """
    # definición de la fecha inicial y final
    fecha_i = fecha - timedelta(hours=delta_hours)
    fecha_f = fecha
    df = df[(df["date"] <= fecha_f) & (df["date"] >= fecha_i)]
    df.reset_index(drop=True, inplace=True)
    return df


def active_time_table(fecha, dias, gps=pd.DataFrame(), loads=pd.DataFrame(),
                      mems=pd.DataFrame()):
    """
    Verificar en una tabla los tiempos que estuvieron encendidos, según
    las distintas fuentes de datos, de manera de llegar a un tiempo
    encendido por fuentes de datos
    Parameters
    ----------
    fecha : datetime
        fecha actual.
    dias : int
        dias hacia atrás a analizar.
        DESCRIPTION.
    loads : dataframe, optional
        dataframe con cargas y descargas de mineral.
        The default is pd.DataFrame().
    gps : dataframe, optional
        gps de los equipos. The default is pd.DataFrame().
    mems : dataframe, optional
        presión y temperatura de los equipos.
        The default is pd.DataFrame().
    Returns
    -------
    tabla : TYPE
        DESCRIPTION.
    calidad : TYPE
        DESCRIPTION.
    """

    lista = []
    gps_on = gps.copy()
    cargas_on = loads.copy()
    mems_on = mems.copy()

    fecha_final = fecha.strftime("%m/%d/%Y")
    fecha_inicio = (fecha - timedelta(days=dias)).strftime("%m/%d/%Y")
    # lista de fechas original
    index = pd.date_range(start=fecha_inicio, end=fecha_final,
                          freq="min").values
    fechas = pd.DataFrame(index)
    fechas["on"] = 0
    fechas.columns = ["date", "on"]
    fechas["date"] =\
        fechas["date"].apply(lambda x: timezone_fechas("America/Santiago", x))
    fechas["date"] = fechas["date"].astype('datetime64[ns, America/Santiago]')

    # Redondeamos las fechas de cargas
    if len(cargas_on) > 0:
        cargas_on["date"] =\
            cargas_on["date"].apply(lambda x: round_1min(x))
        cargas_on.apply(get_dates_loads, args=(lista,), axis=1)
        # Nueva lista
        fechas_cargas = pd.DataFrame(lista, columns=["date", "equipo", "on"])
        fechas_cargas["date"] = fechas_cargas["date"].dt.tz_localize('utc') \
            .dt.tz_convert('America/Santiago')
        fechas = pd.concat([fechas, fechas_cargas])
        fechas.drop_duplicates(subset=["date", "equipo"], inplace=True)

    if len(gps_on) > 0:
        gps_on["date"] = gps_on["date"].\
            apply(lambda x: round_1min(x))
        gps_on = gps_on[gps_on["date"] != 0]
        gps_on.reset_index(inplace=True, drop=True)
        gps_on["on"] = 1
        fechas = pd.concat([fechas, gps_on[["date", "equipo", "on"]]])
        fechas.drop_duplicates(subset=["date", "equipo"], inplace=True)

    if len(mems_on) > 0:
        mems_on["date"] = mems_on["date"].\
            apply(lambda x: round_1min(x))
        mems_on = mems_on[mems_on["date"] != 0]
        mems_on.reset_index(inplace=True, drop=True)
        mems_on["on"] = 1
        fechas = pd.concat([fechas, mems_on[["date", "equipo", "on"]]])
        fechas.drop_duplicates(subset=["date", "equipo"], inplace=True)

    tabla = pd.pivot_table(fechas, index="date", columns="equipo", values="on")
    tabla.fillna(0, inplace=True)
    tabla.reset_index(inplace=True)
    return tabla


def gps_quality(fecha, dias, gps=pd.DataFrame(), loads=pd.DataFrame(),
                mems=pd.DataFrame()):
    """
    Verificar en una tabla los tiempos que estuvieron encendidos, según
    las distintas fuentes de datos, de manera de llegar a un tiempo
    encendido por fuentes de datos
    Parameters
    ----------
    fecha : datetime
        fecha actual.
    dias : int
        dias hacia atrás a analizar.
        DESCRIPTION.
    loads : dataframe, optional
        dataframe con cargas y descargas de mineral.
        The default is pd.DataFrame().
    gps : dataframe, optional
        gps de los equipos. The default is pd.DataFrame().
    mems : dataframe, optional
        presión y temperatura de los equipos.
        The default is pd.DataFrame().
    Returns
    -------
    tabla : TYPE
        DESCRIPTION.
    calidad : TYPE
        DESCRIPTION.
    """

    lista = []
    gps_on = gps.copy()
    cargas_on = loads.copy()
    mems_on = mems.copy()

    fecha_final = fecha.strftime("%m/%d/%Y")
    fecha_inicio = (fecha - timedelta(days=dias)).strftime("%m/%d/%Y")
    # lista de fechas original
    index = pd.date_range(start=fecha_inicio, end=fecha_final,
                          freq="min").values
    fechas = pd.DataFrame(index)
    fechas["on"] = 0
    fechas.columns = ["date", "on"]
    fechas["date"] =\
        fechas["date"].apply(lambda x: timezone_fechas("America/Santiago", x))
    fechas["date"] = fechas["date"].astype('datetime64[ns, America/Santiago]')

    # Redondeamos las fechas de cargas
    if len(cargas_on) > 0:
        cargas_on["date"] =\
            cargas_on["date"].apply(lambda x: round_1min(x))
        cargas_on.apply(get_dates_loads, args=(lista,), axis=1)
        # Nueva lista
        fechas_cargas = pd.DataFrame(lista, columns=["date", "equipo", "on"])
        fechas_cargas["date"] = fechas_cargas["date"].dt.tz_localize('utc') \
            .dt.tz_convert('America/Santiago')
        fechas = pd.concat([fechas, fechas_cargas])
        fechas.drop_duplicates(subset=["date", "equipo"], inplace=True)

    if len(gps_on) > 0:
        gps_on["date"] = gps_on["date"].\
            apply(lambda x: round_1min(x))
        gps_on = gps_on[gps_on["date"] != 0]
        gps_on.reset_index(inplace=True, drop=True)
        gps_on["on"] = 1
        fechas = pd.concat([fechas, gps_on[["date", "equipo", "on"]]])
        fechas.drop_duplicates(subset=["date", "equipo"], inplace=True)

    if len(mems_on) > 0:
        mems_on["date"] = mems_on["date"].\
            apply(lambda x: round_1min(x))
        mems_on = mems_on[mems_on["date"] != 0]
        mems_on.reset_index(inplace=True, drop=True)
        mems_on["on"] = 1
        fechas = pd.concat([fechas, mems_on[["date", "equipo", "on"]]])
        fechas.drop_duplicates(subset=["date", "equipo"], inplace=True)

    tabla = pd.pivot_table(fechas, index="date", columns="equipo", values="on")
    tabla.fillna(0, inplace=True)
    tabla.reset_index(inplace=True)
    # Se agrega la verificacion de calidad de gps
    cargas_dates = fechas_cargas.copy()
    cargas_dates = pd.pivot_table(cargas_dates, index="date",
                                  columns="equipo", values="on")
    cargas_dates = cargas_dates.replace(1, 99)
    # Las fechas en que hay gps de un camion
    if len(gps_on) > 0:
        gps_dates = gps_on[["date", "equipo", "on"]].copy()
        gps_dates = pd.pivot_table(gps_dates, index="date",
                                   columns="equipo", values="on")
        # Los sumo, si es 100 hay cargas y gps, si es 99 solo hay cargas, si
        # es 1 solo hay gps
        calidad = pd.concat([cargas_dates, gps_dates]).reset_index()
    else:
        calidad = cargas_dates.copy()
    calidad = calidad.groupby("date").sum().reset_index()
    return tabla, calidad


def truck_load_status(gps, loads):
    """
    Poner en gps el estado de carga de mineral en la base de datos de gps

    Parameters
    ----------
    gps : dataframe
        datos de gps.
    loads : dataframe
        datos de cargas y descargas.
    Returns
    -------
    output : dataframe
        gps con estado de carga de mineral y el tonelaje transportado.
    """
    output = pd.DataFrame()
    for equipo in gps["equipo"].unique():
        gps_camion = gps[gps["equipo"] == equipo]
        cargas_camion = loads[loads["equipo"] == equipo]
        gps_camion.reset_index(drop=True, inplace=True)
        cargas_camion.reset_index(drop=True, inplace=True)
        for carga in range(len(cargas_camion)):
            gps_camion.loc[(cargas_camion.loc[carga, "date"] <
                            gps_camion["date"]) &
                           (cargas_camion.loc[carga,
                                              "fecha_fin_descarga"] >
                            gps_camion["date"]), "estado_carga"] = "cargado"

            gps_camion.loc[(cargas_camion.loc[carga, "date"] <
                            gps_camion["date"]) &
                           (cargas_camion.loc[carga,
                                              "fecha_fin_descarga"] >
                            gps_camion["date"]), "tonelaje"] =\
                cargas_camion.loc[carga, "tonelaje"]

            gps_camion["estado_carga"].fillna("vacio", inplace=True)
            gps_camion["tonelaje"].fillna(0, inplace=True)
        output = pd.concat([output, gps_camion], axis=0)
    output.sort_values(by=["equipo", "date"], inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output


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


def friction_coefficient(velocidad, coef_estatico, coef_dinamico, velim=3):
    """
    Según la velocidad del equipo, se le asigna un coeficiente de roce,
    velim es el parámetro que determina cuando es coeficiente de roce
    dinamico y cuando es estatico.
    Parameters
    ----------
    velocidad : float
        velocidad en un instante t.
    coef_estatico : float
        coeficiente de roce estatico.
    coef_estatico : float
        coeficiente de roce dinamico.
    velim : float, optional
        por sobre esta velocidad es coeficiente de roce dinamico.
        The default is 3.
    Returns
    -------
    salida : float
        coeficiente de roce en función de su velocidad.
    """
    if velocidad > velim:
        salida = coef_dinamico
    else:
        salida = coef_estatico
    return salida


def autocorrelation_series(df, lag=1):
    """
    Retorna la autocorrelación de un de un valor con su precedente
    Parameters
    ----------
    df : dataframe
        dataframe de las series que se quiere sacar la autocorrelación.
    lag : int, optional
        valores lag hacia atrás de lag.
        The default is 1.
    Returns
    -------
    salida : dataframe
        autocorrelaciones de cada columna agrupadas por equipo.
    """
    salida = pd.DataFrame(df["equipo"].unique())
    salida.columns = ["equipo"]
    for col in df.columns:
        new_col = f"ac_{lag}_" + col
        if "equipo" in col:
            pass
        else:
            valor = pd.DataFrame(
                df.groupby('equipo')[col].apply(pd.Series.autocorr, lag=lag))
            valor.fillna(value=0, inplace=True)
            valor.columns = [new_col]
            valor.reset_index(drop=False, inplace=True)
            salida = salida.merge(valor, on="equipo", how="inner")
    return salida


def normal_force(normal, estado_carga,
                 coef_delantero_descargado,
                 coef_trasero_descargado,
                 coef_delantero_cargado,
                 coef_trasero_cargado):
    """
    Cacular las componentes normales de cada tren del camion, al peso
    que deben soportar, en el estado de equilibrio, lo cual no siempre ocurre

    Parameters
    ----------
    normal : series
        fuerza normal total.
    estado_carga : series
        estado de carga en gps, si esta cargado o vacio.
    coef_delantero_descargado : float
        coeficientes de distribución de peso en estado descargado.
    coef_trasero_descargado : float
        coeficientes de distribución de peso en estado descargado.
    coef_delantero_cargado : float
        coeficientes de distribución de peso en estado cargado.
    coef_trasero_cargado : float
        coeficientes de distribución de peso en estado cargado.
    Returns
    -------
    fuerza_delantera : series
        fuerza en el tren delantero en condiciones de equilibrio en el eje
        XY del camino.
    fuerza_trasera : TYPE
        fuerza en el tren trasero en condiciones de equilibrio en el eje
        XY del camino.
    """
    # en el caso que el camión vaya descargado
    if estado_carga == "vacio":
        fuerza_delantera = coef_delantero_descargado * normal
        fuerza_trasera = coef_trasero_descargado * normal
    # en el caso que el camión vaya cargado
    else:
        fuerza_delantera = coef_delantero_cargado * normal
        fuerza_trasera = coef_trasero_cargado * normal
    return fuerza_delantera, fuerza_trasera


def get_active_time(df, camion):
    """
    Sacar tiempo encendido
    Parameters
    ----------
    df : dataframe
        dataframe de tiempos encendidos.
    camion : string
        camion que se está analizando.
    Returns
    -------
    t_encendido : float
        tiempo encendido del camión según esa fuente de datos.
        Tiempo encendido en horas, dado que el dataframe esta en minutos
    """
    try:
        t_encendido = df[camion].sum() / 60
    except KeyError:
        t_encendido = 0
    return t_encendido


def get_gps_quality(calidad, equipo):
    """
    Determina en un porcentaje de 0-1 la calidad con la que se encuentra
    el gps
    Parameters
    ----------
    calidad : dataframe
        calidad de gps obetenida en procesos anteriores.
    equipo : string
        camion.
    Returns
    -------
    gps_truck_quality : int
        0-1 calidad del gps.
    """
    try:
        gps_truck_status = calidad[[equipo]]
        try:
            gps_truck_quality =\
                ((gps_truck_status[equipo] == 100).sum() /
                 ((gps_truck_status[equipo] == 99) |
                  (gps_truck_status[equipo] == 100)).sum())
        # Si la division es por cero es que no hay registros
        except ZeroDivisionError:
            gps_truck_quality = 1
    # Si el camion no es parte del registro es porque no tiene datos de
    # cargas y por ende no se esta usando
    except KeyError:
        gps_truck_quality = 1
    return gps_truck_quality


def count_database_records(gps=pd.DataFrame(columns=["equipo", "date"]),
                           loads=pd.DataFrame(columns=["equipo", "date"]),
                           mems=pd.DataFrame(columns=["equipo", "date"])):
    """
    Contar los registros por camión de cada uno de los equipos
    Parameters
    ----------
    gps : dataframe, optional
        datos de gps.
        The default is pd.DataFrame(columns=["equipo", "date"]).
    loads : dataframe, optional
        datos de cargas y descargas de mineral.
        The default is pd.DataFrame(columns=["equipo", "date"]).
    mems : dataframe, optional
        datos de presion y temperatura de neumaticos.
        The default is pd.DataFrame(columns=["equipo", "date"]).
    Returns
    -------
    result : dataframe
        conteo de los registros de cada base de datos por camión.

    """
    gps_count = gps.groupby("equipo").count()[["date"]]
    gps_count.columns = ["gps"]
    gps_count.reset_index(drop=False, inplace=True)
    loads_count = loads.groupby("equipo").count()[["date"]]
    loads_count.columns = ["loads"]
    loads_count.reset_index(drop=False, inplace=True)
    mems_count = mems.groupby("equipo").count()[["date"]]
    mems_count.columns = ["mems"]
    mems_count.reset_index(drop=False, inplace=True)
    result = gps_count.merge(loads_count, on="equipo", how="outer")
    result = result.merge(mems_count, on="equipo", how="outer")
    result.fillna(value=0, inplace=True)
    result = rename_columns(result, prefix="conteo")
    result.reset_index(drop=True, inplace=True)
    return result


def base_columns(string):
    """
    Hace un reemplazo para ser usado en un apply de lambda
    Parameters
    ----------
    string : string
        string con el nombre de la columna.
    Returns
    -------
    string
        string filtrado sin las palabras delatera y trasera.
    """
    return string.replace("_delantera", "").replace("_trasera", "")


def back_front_metric(df):
    """
    Retorna las columnas asociadas a metricas de fuerzas delanteras
    o traseras
    Parameters
    ----------
    df : dataframe
        merge entre las metricas por camión y las metricas por neumático
        de mems.
    Returns
    -------
    colsdt : dataframe
        dataframe de las columnas que están asociadas a una
        posición de la rueda, como columnas únicas.
    """
    alpha = []
    for col in df.columns:
        if "delantera" in col or "trasera" in col:
            if "neumatico" in col:
                alpha.append(col)
    colsdt = pd.DataFrame(alpha, columns=["columnas"])
    colsdt["base"] = colsdt["columnas"].apply(str)
    colsdt["base"] = colsdt["base"].apply(lambda x: base_columns(x))
    colsdt = colsdt[["base"]].drop_duplicates()
    colsdt.reset_index(drop=True, inplace=True)
    return colsdt


def get_tires_position(
        mems=pd.DataFrame(columns=["equipo", "key_neumatico"])):
    """
    Asigna un posición de cada neumático con la lógica de:
        El últiumo tercio de vida útil se hace en la parte delantera por tanto
        los más nuevos empiezan en la parte trasera
    Parameters
    ----------
    mems : dataframe, optional
        datos de mems.
        The default is pd.DataFrame(columns=["equipo", "key_neumatico"]).
    Returns
    -------
    posicion_neumaticos : dataframe
        dataframe por id de neumático y camión asignada una posición.
    """

    neumaticos = mems.groupby(["equipo", "key_neumatico"]).mean()
    neumaticos.reset_index(drop=False, inplace=True)
    neumaticos = neumaticos[["equipo", "key_neumatico"]]
    neumaticos.sort_values(by=["equipo", "key_neumatico"],
                           inplace=True)
    posicion_neumaticos = pd.DataFrame()
    for equipo in neumaticos["equipo"].unique():
        neumaticos_e = neumaticos[neumaticos["equipo"] == equipo]
        neumaticos_e.reset_index(drop=True, inplace=True)
        neumaticos_e.reset_index(drop=False, inplace=True)
        neumaticos_e.rename(columns={"index": "posicion"}, inplace=True)
        neumaticos_e["posicion"] = neumaticos_e["posicion"] + 1
        neumaticos_e["posicion"] = neumaticos_e["posicion"].apply(
            lambda x: "delantera" if x >= 5 else "trasera")
        posicion_neumaticos = pd.concat([posicion_neumaticos, neumaticos_e],
                                        axis=0)
    posicion_neumaticos.reset_index(drop=True, inplace=True)
    posicion_neumaticos = posicion_neumaticos[["equipo", "key_neumatico",
                                               "posicion"]]
    return posicion_neumaticos


def selecting_position(delantera, trasera, posicion):
    """
    En función de una colunmna dada, retorna el esta tiene fuerzas equivalentes
    a delanteras o traseras
    Parameters
    ----------
    delantera : float
        columna delantera de alguna metrica.
    trasera : float
        columna delantera de alguna metrica.
    posicion : string
        posición de la rueda.
    Returns
    -------
    salida : float
        Elección de alguna de las columnas.
    """
    if posicion == "delantera":
        salida = delantera
    else:
        salida = trasera
    return salida


def compute_ads(fecha, dias, gps=pd.DataFrame(), loads=pd.DataFrame,
                mems=pd.DataFrame(), delta_hours=24):
    """
    Hace el computo del ads por camión

    Parameters
    ----------
    fecha : datetime
        fecha actual.
    dias : int
        dias hacia atrás para generar ads.
    gps : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    loads : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame.
    mems : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().
    delta_hours : int, optional
        Es la cantidad de horas hacia atrás que tendrá el ads.
        The default is 24.
    Returns
    -------
    None.

    """
    # definición de parámetros básicos

    # cliente ssm
    ssm = boto3.client("ssm")
    # coeficiente de roce estatico
    coef_estatico = float(ssm.get_parameter(
        Name='CoeficienteRoceEstatico',
        WithDecryption=True)["Parameter"]["Value"])
    # coeficiente de roce dinámico
    coef_dinamico = float(ssm.get_parameter(
        Name='CoeficienteRoceDinamico',
        WithDecryption=True)["Parameter"]["Value"])
    # coeficiente de peso delantero en estado descargado
    coef_delantero_descargado = float(ssm.get_parameter(
        Name='PesoDelanteroDescargado',
        WithDecryption=True)["Parameter"]["Value"])
    # coeficiente de peso trasero en estado descargado
    coef_trasero_descargado = float(ssm.get_parameter(
        Name='PesoTraseroDescargado',
        WithDecryption=True)["Parameter"]["Value"])
    # coeficiente de peso delantero en estado cargado
    coef_delantero_cargado = float(ssm.get_parameter(
        Name='PesoDelanteroCargado',
        WithDecryption=True)["Parameter"]["Value"])
    # coeficiente de peso atrás en estado cargado
    coef_trasero_cargado = float(ssm.get_parameter(
        Name='PesoTraseroCargado',
        WithDecryption=True)["Parameter"]["Value"])
    # aceleración de gravedad [m / s2]
    ag = 9.81
    # limite de tiempo en segundos que no puede ser mayor
    epsilon_t = 60 * 30

    # ver el estado de carga en gps
    gps = truck_load_status(gps, loads)

    # definir estado de pendiente
    gps["estado_carga"].fillna(value="vacio", inplace=True)
    gps["estado_pendiente"].fillna(value="plano", inplace=True)
    gps["estado_global"] = gps["estado_pendiente"] + " " + gps["estado_carga"]
    # masa base del camión
    gps['masa_base'] = 210.187 * 1000
    # calculo del peso total del camión en [kg]
    gps['masa_total'] = gps['tonelaje'] * 1000 + gps['masa_base']
    # calculo del coeficiente de roce en función de la velocidad
    gps["coeficiente_roce"] = gps["velocidad_estimada"].apply(
        lambda x: friction_coefficient(
            x, coef_estatico, coef_dinamico, velim=3))

    # fuerza debido a aceleracion imprimida en el cuerpo
    # fuerza de aceleracion F = M * a
    gps["f_newton"] = gps["masa_total"] * gps["aceleracion"]
    gps["f_newton_positiva"] = gps["masa_total"] * \
        gps["aceleracion_positiva"]
    gps["f_newton_neagtiva"] = gps["masa_total"] * \
        gps["aceleracion_negativa"]

    # fuerza motriz
    gps["f_motriz"] = gps["masa_total"] * (
        gps["aceleracion"] + ag * (
            gps["coeficiente_roce"] * np.cos(gps["angulo_rad"]) + np.sin(
                gps["angulo_rad"])))
    # fuerza motriz solo positiva
    gps["f_motriz_positiva"] = gps["f_motriz"].apply(
        lambda x: 0 if x < 0 else x)
    # fuerza motriz negativa
    gps["f_motriz_negativa"] = gps["f_motriz"].apply(
        lambda x: 0 if x > 0 else x)

    # fuerza normal del peso total
    gps["fn_total"] = ag * gps['masa_total'] * np.cos(
        gps["angulo_rad"])
    # fuerza normal tren delantero y tren trasero por neumatico
    gps[["fn_delantera", "fn_trasera"]] = gps.apply(
        lambda x: normal_force(
            x["fn_total"],
            x["estado_carga"],
            coef_delantero_descargado, coef_trasero_descargado,
            coef_delantero_cargado, coef_trasero_cargado), axis=1,
        result_type="expand")

    # calculando las fuerzas de roce
    gps["fr_delantera"] = gps["fn_delantera"] * gps["coeficiente_roce"]
    gps["fr_trasera"] = gps["fn_trasera"] * gps["coeficiente_roce"]

    # a nivel de neumatico en cada tren, fuerza normal por neumatico
    gps["fn_neumatico_delantera"] = gps["fn_delantera"] / 2
    gps["fn_neumatico_trasera"] = gps["fn_trasera"] / 4

    # a nivel de neumatico en cada tren, fuerza de roce por neumatico
    gps["fr_neumatico_delantera"] = gps["fn_delantera"] * \
        gps["coeficiente_roce"] / 2
    gps["fr_neumatico_trasera"] = gps["fn_trasera"] * \
        gps["coeficiente_roce"] / 4

    # trabajo mecánico realizado por el camión
    gps["w_neto"] = gps["f_motriz"] * gps["movimiento"]
    # trabajo neto en direccion positiva
    gps["w_neto_positivo"] = gps["w_neto"].apply(lambda x: 0 if x < 0 else x)
    # trabajo neto en direccion de frenado
    gps["w_neto_negativo"] = gps["w_neto"].apply(lambda x: 0 if x > 0 else x)

    # trabajo neto realizado en el tren delantero
    gps["wfr_delantera"] = gps["fr_delantera"] * gps["movimiento"]
    gps["wfr_trasera"] = gps["fr_trasera"] * gps["movimiento"]

    # trabajo mecánico realizado por el roce en las ruedas delanteras
    gps["wfr_neumatico_delantera"] = gps[
        "fr_neumatico_delantera"] * gps["movimiento"]
    # trabajo mecánico realizado por el roce en las ruedas delanteras
    gps["wfr_neumatico_trasera"] = gps[
        "fr_neumatico_trasera"] * gps["movimiento"]

    # filtros del ultimo dia solamente
    gps = filter_by_hours(gps, fecha, delta_hours)
    loads = filter_by_hours(loads, fecha, delta_hours)
    mems = filter_by_hours(mems, fecha, delta_hours)

    print("Fechas de gps:", gps["date"].min(), gps["date"].max())
    print("Fechas de loads:", loads["date"].min(), loads["date"].max())
    print("Fechas de mems:", mems["date"].min(), mems["date"].max())

    # sacando metricas de las series temporales
    columnas_series = ['equipo', 'angulo', 'angulo_positivo',
                       'angulo_negativo', 'aceleracion',
                       'aceleracion_positiva', 'aceleracion_negativa',
                       'f_newton', 'f_newton_positiva', 'f_newton_neagtiva',
                       'f_motriz', 'f_motriz_positiva', 'f_motriz_negativa',
                       'fn_total', 'fn_delantera', 'fn_trasera',
                       'fr_delantera', 'fr_trasera', 'fn_neumatico_delantera',
                       'fn_neumatico_trasera', 'fr_neumatico_delantera',
                       'fr_neumatico_trasera', 'w_neto', 'w_neto_positivo',
                       'w_neto_negativo', 'wfr_delantera', 'wfr_trasera',
                       'wfr_neumatico_delantera', 'wfr_neumatico_trasera']
    # variables de suma
    suma = gps[columnas_series]
    suma = suma.groupby("equipo").sum()
    suma.reset_index(drop=False, inplace=True)
    suma = rename_columns(suma, prefix="sum")

    # variables promedio
    promedio = gps[columnas_series]
    promedio = promedio.groupby("equipo").mean()
    promedio.reset_index(drop=False, inplace=True)
    promedio = rename_columns(promedio, prefix="mean")

    # desviación estandar
    desviacion = gps[columnas_series]
    desviacion = desviacion.groupby("equipo").std()
    desviacion.reset_index(drop=False, inplace=True)
    desviacion = rename_columns(desviacion, prefix="std")

    # variables de mediana
    mediana = gps[columnas_series]
    mediana = mediana.groupby("equipo").median()
    mediana.reset_index(drop=False, inplace=True)
    mediana = rename_columns(mediana, prefix="median")

    # conteo de registros por base de datos
    count = count_database_records(gps, loads, mems)

    # kurtosis de cada variable
    kurtosis = gps[columnas_series]
    kurtosis = kurtosis.groupby('equipo').apply(pd.DataFrame.kurt)
    kurtosis.reset_index(drop=False, inplace=True)
    kurtosis = rename_columns(kurtosis, prefix="kurt")

    # skewness de cada variable
    skewness = gps[columnas_series]
    skewness = skewness.groupby('equipo').apply(pd.DataFrame.kurt)
    skewness.reset_index(drop=False, inplace=True)
    skewness = rename_columns(skewness, prefix="skewness")

    # autocorrelación
    autocorr = gps[columnas_series]
    autocorr.reset_index(drop=True, inplace=True)
    autocorr1 = autocorrelation_series(autocorr, lag=1)
    autocorr2 = autocorrelation_series(autocorr, lag=2)
    autocorr3 = autocorrelation_series(autocorr, lag=3)

    # merge de resultados por camión
    result = suma.merge(promedio, on="equipo", how="outer")
    result = result.merge(desviacion, on="equipo", how="outer")
    result = result.merge(mediana, on="equipo", how="outer")
    result = result.merge(count, on="equipo", how="outer")
    result = result.merge(kurtosis, on="equipo", how="outer")
    result = result.merge(skewness, on="equipo", how="outer")
    result = result.merge(autocorr1, on="equipo", how="outer")
    result = result.merge(autocorr2, on="equipo", how="outer")
    result = result.merge(autocorr3, on="equipo", how="outer")

    # tiempo activo de los equipos según las distintas fuentes de datos
    # tiempo activo según gps
    active_time_gps = active_time_table(
        fecha, dias, gps=gps, loads=pd.DataFrame(), mems=pd.DataFrame())
    # tiempo activo según cargas y descargas de mineral
    active_time_loads = active_time_table(
        fecha, dias, gps=pd.DataFrame(), loads=loads, mems=pd.DataFrame())
    # tiempo activo según sistema mems
    active_time_mems = active_time_table(
        fecha, dias, gps=pd.DataFrame(), loads=pd.DataFrame(), mems=mems)
    # calidad de gps más tiempo total activo, según todas las fuentes
    active_time_all, calidad = gps_quality(
        fecha, dias, gps, loads, mems)

    # equipos existentes
    equipos = list(active_time_all.columns)
    # equipos.remove("date")

    # obtención de metricas de mems

    # presión y temperatura suma
    mems_sum = mems.groupby(["equipo", "key_neumatico"]).sum()
    mems_sum.reset_index(drop=False, inplace=True)
    mems_sum = rename_columns(mems_sum, prefix="sum")

    # presión y temperatura medias
    mems_mean = mems.groupby(["equipo", "key_neumatico"]).mean()
    mems_mean.reset_index(drop=False, inplace=True)
    mems_mean = rename_columns(mems_mean, prefix="mean")

    # desviación estadar
    mems_std = mems.groupby(["equipo", "key_neumatico"]).std()
    mems_std.reset_index(drop=False, inplace=True)
    mems_std = rename_columns(mems_std, prefix="std")

    # conteo de valores
    mems_count = mems.groupby(["equipo", "key_neumatico"]).count()[["date"]]
    mems_count.columns = ["mems_neumatico"]
    mems_count.reset_index(drop=False, inplace=True)
    mems_count = rename_columns(mems_count, prefix="count")

    # curtosis del parámetro
    mems_kurtosis = mems.groupby(
        ["equipo", "key_neumatico"]).apply(pd.DataFrame.kurt)
    # mems_kurtosis.drop(columns=['date', 'key_neumatico'], inplace=True)
    mems_kurtosis.drop(columns=['key_neumatico'], inplace=True)
    mems_kurtosis.reset_index(drop=False, inplace=True)
    mems_kurtosis = rename_columns(mems_kurtosis, prefix="kurtosis")

    # skewnees sesgo de la muestras
    mems_skewness = mems.groupby(
        ["equipo", "key_neumatico"]).apply(pd.DataFrame.skew)
    # mems_skewness.drop(columns=['date', 'key_neumatico'], inplace=True)
    mems_skewness.drop(columns=['key_neumatico'], inplace=True)
    mems_skewness.reset_index(drop=False, inplace=True)
    mems_skewness = rename_columns(mems_skewness, prefix="skewness")

    metricas_mems = mems_sum.merge(
        mems_mean, on=["equipo", "key_neumatico"], how="outer")
    metricas_mems = metricas_mems.merge(
        mems_std, on=["equipo", "key_neumatico"], how="outer")
    metricas_mems = metricas_mems.merge(
        mems_count, on=["equipo", "key_neumatico"], how="outer")
    metricas_mems = metricas_mems.merge(
        mems_kurtosis, on=["equipo", "key_neumatico"], how="outer")
    metricas_mems = metricas_mems.merge(
        mems_skewness, on=["equipo", "key_neumatico"], how="outer")

    # iterar por equipos
    equipos.remove("date")

    metricas = []
    for equipo in equipos:
        print("Calculando para el equipo: ", equipo, "...")
        gps_e = gps[gps["equipo"] == equipo]
        loads_e = loads[loads["equipo"] == equipo]
        mems_e = mems[mems["equipo"] == equipo]

        # reseteo de index
        gps_e.reset_index(drop=True, inplace=True)
        loads_e.reset_index(drop=True, inplace=True)
        mems_e.reset_index(drop=True, inplace=True)

        # toneladas transportadas
        toneladas_transportadas = loads_e["tonelaje"].sum()
        toneladas_por_viaje = loads_e["tonelaje"].sum() / len(loads_e)

        # metrica de calidad de gps
        calidad_equipo = get_gps_quality(calidad, equipo)

        # calculo de tiempos encendidos según las distintas fuentes de datos
        t_encendido = get_active_time(active_time_all, equipo)
        t_encendido_gps = get_active_time(active_time_gps, equipo)
        t_encendido_loads = get_active_time(active_time_loads, equipo)
        t_encendido_mems = get_active_time(active_time_mems, equipo)

        # tiempo en los distintos estados de la mina en [hr]
        t_subiendo_c = gps_e[(gps_e[
            "estado_global"] == "subiendo cargado") &
            (gps_e["diff_date"] <= epsilon_t)]["diff_date"].sum() / 3600
        t_subiendo_v =\
            gps_e[(gps_e["estado_global"] ==
                   "subiendo vacio") &
                  (gps_e["diff_date"] <= epsilon_t)]["diff_date"].sum() / 3600
        t_bajando_c =\
            gps_e[(gps_e["estado_global"] ==
                   "bajando cargado") &
                  (gps_e["diff_date"] <= epsilon_t)]["diff_date"].sum() / 3600
        t_bajando_v =\
            gps_e[(gps_e["estado_global"] ==
                   "bajando vacio") &
                  (gps_e["diff_date"] <= epsilon_t)]["diff_date"].sum() / 3600
        t_plano_c =\
            gps_e[(gps_e["estado_global"] ==
                   "plano cargado") &
                  (gps_e["diff_date"] <= epsilon_t)]["diff_date"].sum() / 3600
        t_plano_v =\
            gps_e[(gps_e["estado_global"] ==
                   "plano vacio") &
                  (gps_e["diff_date"] <= epsilon_t)]["diff_date"].sum() / 3600

        # metricas de distancias en [km]
        dist_subiendo_c =\
            gps_e[(gps_e["estado_global"] ==
                   "subiendo cargado")]["movimiento"].sum() / 1000
        dist_subiendo_v =\
            gps_e[(gps_e["estado_global"] ==
                   "subiendo vacio")]["movimiento"].sum() / 1000
        dist_bajando_c =\
            gps_e[(gps_e["estado_global"] ==
                   "bajando cargado")]["movimiento"].sum() / 1000
        dist_bajando_v =\
            gps_e[(gps_e["estado_global"] ==
                   "bajando vacio")]["movimiento"].sum() / 1000
        dist_plano_c =\
            gps_e[(gps_e["estado_global"] ==
                   "plano cargado")]["movimiento"].sum() / 1000
        dist_plano_v =\
            gps_e[(gps_e["estado_global"] ==
                   "plano vacio")]["movimiento"].sum() / 1000

        # velocidades en [km/ hr]
        v_subiendo_c =\
            gps_e[gps_e["estado_global"] ==
                  "subiendo cargado"]["velocidad_estimada"].mean()
        v_subiendo_v =\
            gps_e[gps_e["estado_global"] ==
                  "subiendo vacio"]["velocidad_estimada"].mean()
        v_bajando_c =\
            gps_e[gps_e["estado_global"] ==
                  "bajando cargado"]["velocidad_estimada"].mean()
        v_bajando_v =\
            gps_e[gps_e["estado_global"] ==
                  "bajando vacio"]["velocidad_estimada"].mean()
        v_plano_c =\
            gps_e[gps_e["estado_global"] ==
                  "plano cargado"]["velocidad_estimada"].mean()
        v_plano_v =\
            gps_e[gps_e["estado_global"] ==
                  "plano vacio"]["velocidad_estimada"].mean()

        # velocidades en [km/ hr] solo mayores a 5 km / hr
        # calculo de las velocidades en movimiento de la mina
        lim_vel = 5
        vr_subiendo_c =\
            gps_e[
                (gps_e["estado_global"] == "subiendo cargado") &
                (gps["velocidad_estimada"] > lim_vel)][
                    "velocidad_estimada"].mean()
        vr_subiendo_v =\
            gps_e[(gps_e["estado_global"] == "subiendo vacio") &
                  (gps["velocidad_estimada"] > lim_vel)][
                      "velocidad_estimada"].mean()
        vr_bajando_c =\
            gps_e[(gps_e["estado_global"] == "bajando cargado") &
                  (gps["velocidad_estimada"] > lim_vel)][
                      "velocidad_estimada"].mean()
        vr_bajando_v =\
            gps_e[(gps_e["estado_global"] == "bajando vacio") &
                  (gps["velocidad_estimada"] > lim_vel)][
                      "velocidad_estimada"].mean()
        vr_plano_c =\
            gps_e[(gps_e["estado_global"] ==
                   "plano cargado") &
                  (gps["velocidad_estimada"] > lim_vel)][
                      "velocidad_estimada"].mean()
        vr_plano_v =\
            gps_e[(gps_e["estado_global"] == "plano vacio") &
                  (gps["velocidad_estimada"] > lim_vel)][
                      "velocidad_estimada"].mean()

        # ir guardando las metricas que en una lista
        metricas.append([equipo, fecha, toneladas_transportadas,
                         toneladas_por_viaje, calidad_equipo, t_encendido,
                         t_encendido_gps, t_encendido_loads,
                         t_encendido_mems, t_subiendo_c, t_subiendo_v,
                         t_bajando_c, t_bajando_v, t_plano_c, t_plano_v,
                         dist_subiendo_c, dist_subiendo_v, dist_bajando_c,
                         dist_bajando_v, dist_plano_c, dist_plano_v,
                         v_subiendo_c, v_subiendo_v, v_bajando_c,
                         v_bajando_v, v_plano_c, v_plano_v,
                         vr_subiendo_c, vr_subiendo_v, vr_bajando_c,
                         vr_bajando_v, vr_plano_c, vr_plano_v])
    metricas = pd.DataFrame(
        metricas, columns=["equipo", "fecha", "toneladas_transportadas",
                           "toneladas_por_viaje", "calidad_equipo",
                           "t_encendido", "t_encendido_gps",
                           "t_encendido_loads", "t_encendido_mems",
                           "t_subiendo_c", "t_subiendo_v", "t_bajando_c",
                           "t_bajando_v", "t_plano_c", "t_plano_v",
                           "dist_subiendo_c", "dist_subiendo_v",
                           "dist_bajando_c", "dist_bajando_v",
                           "dist_plano_c", "dist_plano_v", "v_subiendo_c",
                           "v_subiendo_v", "v_bajando_c", "v_bajando_v",
                           "v_plano_c", "v_plano_v", "vr_subiendo_c",
                           "vr_subiendo_v", "vr_bajando_c", "vr_bajando_v",
                           "vr_plano_c", "vr_plano_v"])

    # resultado final del ads sin metricas de mems
    metricas = metricas.merge(result, on="equipo", how="outer")
    metricas.fillna(value=0, inplace=True)
    metricas.reset_index(drop=True, inplace=True)

    # A NIVEL DE NEUMÁTICOS
    # neumaticos existentes con la posición en función de la logica minera
    neumaticos = get_tires_position(mems)
    metricas_mems = metricas_mems.merge(
        neumaticos, on=["equipo", "key_neumatico"], how="outer")
    # dataframe de conexión con las metricas de gps y loads
    work = metricas.merge(metricas_mems, on="equipo", how="inner")
    # columnas que poseen la palabra delantera o trasera
    colsdt = back_front_metric(work)
    # iterar en las columnas base
    eliminar = []
    for col in colsdt["base"].unique():
        col_delantera = col + "_delantera"
        col_trasera = col + "_trasera"
        # print(col, ":",  col_delantera, ":", col_trasera)
        work[col] = work[[col_delantera, col_trasera, "posicion"]].apply(
            lambda x: selecting_position(
                x[col_delantera], x[col_trasera], x["posicion"]), axis=1)
        eliminar.append(col_delantera)
        eliminar.append(col_trasera)

    # agregar la fecha
    work.drop(columns=eliminar, inplace=True)
    work.reset_index(drop=True, inplace=True)
    work.fillna(value=0, inplace=True)
    return work


def upload_files_as_pkl(files, fecha):
    """
    La funcion guarda en s3 un pkl de los dataframes entregados
    Parameters
    ----------
    files : list
        contiene el nombre del archivo, el dataframe, prefijo y bucket
    day : int
        dia de la data.
    month : int
        mes de la data
    year : int
        year de la data
    Returns
    -------
    None.
    """
    day = fecha.day
    month = fecha.month
    year = fecha.year
    hour = fecha.hour
    # Conexion a s3
    s3 = boto3.client("s3")
    # Para cada uno de los archivos
    for file in files:
        data_name = file[0]
        prefix = file[2]
        bucket = file[3]
        ruta = (f"{prefix}/{year}/{month}/{day}/{data_name}_"
                f"{day}-{month}-{year}-{hour}.pkl")
        # gaurdar el archivo
        file[1].to_pickle("/tmp/file.pkl")
        s3.upload_file(Filename="/tmp/file.pkl",
                       Bucket=bucket,
                       Key=ruta)
        os.remove("/tmp/file.pkl")
