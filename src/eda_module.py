import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('bmh')


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


def plot_categorical_columns(df_categ, path="MKPF", labelsize_=30):
    """
    Plot de las columnas categoricas como un gráfico de frecuencias
    Parameters
    ----------
    df_categ : TYPE
        DESCRIPTION.
    path : TYPE, optional
        DESCRIPTION. The default is ".png".
    labelsize_ : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    None.

    """
    # plot de frecuencias de los datos
    for col in df_categ.columns:
        df_not_num = df_categ[[col]]
        fig, axes = plt.subplots(
            round(len(df_not_num.columns) / 1), 1, figsize=(30, 20))
        for i, ax in enumerate(fig.axes):
            if i < len(df_not_num.columns):
                column = list(df_not_num.columns)[0]
                ax.set_xticklabels(ax.xaxis.get_majorticklabels(),
                                   rotation=45, fontsize=labelsize_)
                ax.xaxis.set_tick_params(labelsize=labelsize_)
                ax.yaxis.set_tick_params(labelsize=labelsize_)
                ax.set_title('Variable'+column,
                             fontweight="bold", size=labelsize_+10)
                ax.set_ylabel(column, fontsize=labelsize_)
                ax.set_xlabel('Variables', fontsize=labelsize_)
                ax.set_facecolor('white')
                sns.countplot(x=df_not_num.columns[i],
                              alpha=1, data=df_not_num, ax=ax)
                fig.savefig(
                    f"data/output-data/eda/{path}/{column}.png", dpi=400)
        fig.tight_layout()


def exploratory_analysis(filename, porcentaje_aceptacion=30, labelsize_=30):
    """
    Generalización de exploración de datos
    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    porcentaje_aceptacion : TYPE, optional
        DESCRIPTION. The default is 30.
    labelsize_ : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    None.

    """
    path = f"data/input-data/{filename}.csv"
    df = pd.read_csv(path, index_col=0)
    # información del tipo de dato
    df.info()
    # eliminar espacios
    df = drop_spaces_data(df)
    # hacer identificables los vacios
    df = replace_empty_nans(df)
    # conteo de los datos
    count = pd.DataFrame(df.count()).reset_index(drop=False)
    count.columns = ["columna", "conteo"]
    count["porcentaje"] = count["conteo"] / len(df) * 100
    # eliminar columnas sin un porcentaje aceptable
    drop_cols = count[count["porcentaje"] <=
                      porcentaje_aceptacion]["columna"].to_list()
    df.drop(columns=drop_cols, inplace=True)
    # información del tipo de dato
    tipos = list(set(df.dtypes.tolist()))
    print("Los tipos de data disponibles son:", tipos)

    # analizar solo la data númerica
    df_num = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32',
                                       'float16', 'int16'])
    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    # numerical and categorical columnas
    numerical_columns = list(df_num.columns)
    categorical_columns = list(np.setdiff1d(
        list(df.columns), numerical_columns))
    # matriz de correlación datos numericos
    corr = df_num.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],
                cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                annot=True, annot_kws={"size": 8}, square=True)
    # categoricos hacia cualitativo
    df_categ = df[categorical_columns]
    plot_categorical_columns(df_categ, path=filename, labelsize_=labelsize_)
