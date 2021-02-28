import pandas as pd
from pandas_profiling import ProfileReport
from src.eda_module import (drop_spaces_data, replace_empty_nans)


def exploratory_analysis(path, filename,
                         nombre="Empresas Rey",
                         porcentaje_aceptacion=30,
                         dark_mode=True,
                         minimal_mode=True,
                         aditional=True):
    # leer bases de datos
    if ".csv" in path:
        df = pd.read_csv(path, index_col=0)
    elif ".pkl" in path:
        df = pd.read_pickle(path)
    # información del tipo de dato
    print(df.info())
    # eliminar espacios al principio y final
    df = drop_spaces_data(df)
    # hacer identificables los vacios /  donde hay vacios reemplazar por nans
    df = replace_empty_nans(df)
    # conteo de los datos
    count = pd.DataFrame(df.count()).reset_index(drop=False)
    count.columns = ["columna", "conteo"]
    count["porcentaje"] = count["conteo"] / len(df) * 100
    # eliminar columnas sin un porcentaje aceptable
    drop_cols = count[count["porcentaje"] <=
                      porcentaje_aceptacion]["columna"].to_list()
    # eliminar las columnas que no cumplen un cierto porcentaje de aceptación
    df.drop(columns=drop_cols, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if dark_mode:
        type_html = "-black"
    else:
        type_html = ""

    if minimal_mode:
        title_mode = "-no expensive computations"
        mode = title_mode.replace(" ", "-")
    else:
        title_mode = ""
        mode = title_mode.replace(" ", "-")

    titulo = f"Exploratory Analysis: {nombre}-{filename}{title_mode}"
    titulo = titulo.replace(" ", "-").replace("-", " ").title()
    prof = ProfileReport(
        df,
        title=titulo,
        explorative=False,
        minimal=minimal_mode,
        dark_mode=dark_mode)

    # guardar el html
    base = "data/output-data"
    path_output = f'{base}/{filename}-exploratory{type_html}{mode}.html'
    prof.to_file(output_file=path_output)

    print("Las columnas eliminadas son: ", drop_cols,
          f"debido a que sobrepasan un porcentaje de aceptación del {porcentaje_aceptacion} % de valores nulos")
    print("Las columnas restantes son: ", list(df.columns))
