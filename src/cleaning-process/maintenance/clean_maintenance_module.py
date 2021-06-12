import pytz
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


def transform_loads_date(date):
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
