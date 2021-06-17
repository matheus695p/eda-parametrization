import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def make_empty_identifiable(value):
    """
    Parameters
    ----------
    value : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    if value == "":
        output = np.nan
    else:
        output = value
    return output


def replace_empty_nans(df):
    """
    Parameters
    ----------
    df : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    for col in df.columns:
        print("buscando vacios en:", col, "...")
        df[col] = df[col].apply(lambda x: make_empty_identifiable(x))
    return df


def df2float(df):
    """
    Pasa por las columnas tratando de convertirlas a float64
    Parameters
    ----------
    df : dataframe
        df de trabajo.
    Returns
    -------
    df : dataframe
        df con las columnas númericas en float.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(float)
        except Exception as e:
            print(e)
    df.reset_index(drop=True, inplace=True)
    return df


def lower_rename(df):
    """
    Rename nombre de las columnas

    Parameters
    ----------
    df : dataframe
        renombrar el nombre de las columnas.

    Returns
    -------
    df : dataframe
        dataframe con las columnas en minusculas y sin espacios.

    """
    for col in df.columns:
        print(col)
        new_col = col.lower().replace(" ", "_")
        df.rename(columns={col: new_col}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def drop_nan_columns(df, threshold=0):
    """
    Drop columns con una cantidad de nans mayor a la necesaria

    Parameters
    ----------
    df : dataframe
        dataframe original.
    index : int, optional
        Porcentaje con el cual se borran las columnas. The default is 0.

    Returns
    -------
    df : dataframe
        dataframe con la cantidad eliminada de nans.

    """
    nans_recount = pd.DataFrame(df.isna().sum(), columns=["nans_count"])
    nans_recount.reset_index(drop=False, inplace=True)
    nans_recount["percentage"] = nans_recount["nans_count"] / \
        len(df) * 100
    droping_cols = list(
        nans_recount[nans_recount["percentage"] > threshold]["index"])
    df.drop(columns=droping_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def index_date(df):
    """
    Indexar la columna de las fechas
    Parameters
    ----------
    df : dataframe
        dataframe de demandas de stock en el tiempo.
    Returns
    -------
    df : dataframe
        dataframe con la fecha como index.
    """
    df.rename(columns={"fecha": "index"}, inplace=True)
    df.set_index(["index"], inplace=True)
    return df


def downcast_dtypes(df):
    """
    Función super util para bajar la cantidad de operaciones flotante que
    se van a realizar en el proceso de entrenamiento de la red
    Parameters
    ----------
    df : dataframe
        df a disminuir la cantidad de operaciones flotantes.
    Returns
    -------
    df : dataframe
        dataframe con los tipos int16 y float32 como formato número
    """
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def lstm_preparation(array, timesteps=5):
    """
    Preparar los datos para la predicción con la lstm
    Parameters
    ----------
    array : numpy.array
        array.
    timesteps : int, optional
        cantidad de tiemsteps que se harán las predicciones.
        The default is 5.
    Returns
    -------
    x_train : array
        matriz de entrenamiento de las celdas lstm.
    y_train : array
        salida de las celdas.
    """
    x_train = []
    y_train = []
    for i in range(timesteps, array.shape[0]):
        x_train.append(array[i-timesteps:i])
        y_train.append(array[i][0:array.shape[1]])
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    return x_train, y_train


def nn_preparation(array, names, target_col, timesteps=5):
    """
    Hacer la preparación de la red neuronal
    Parameters
    ----------
    array : numpy.array
        array con todas las variables.
    names : list
        nombre de todas las columnas.
    target_col : string or list
        nombre de la/as columna/as target/s.
    timesteps : int, optional
        cantidad de tiemsteps que se harán las predicciones.
        The default is 5.
    Returns
    -------
    x : array
        x en numpy.
    y : array
        target en numpy.
    """
    df = pd.DataFrame(array, columns=names)
    df = df.iloc[timesteps:, :]
    df.reset_index(drop=True, inplace=True)

    if len(target_col) == 1:
        y = df[[target_col]]
    else:
        y = df[target_col]

    x = df.drop(columns=target_col)
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y
