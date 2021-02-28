import pandas as pd
from pandas_profiling import ProfileReport
from src.eda_module import (drop_spaces_data, replace_empty_nans)

# archivo a analizar
files = ["MKPFdata", "MSEGdata", "R_MOVIMIENTOS_CARGAdata"]
for filename in files:
    print(filename)
    path = f"data/input-data/{filename}.csv"
    nombre = filename.replace("data", "")
    path_names = f"data/input-data/{nombre}.csv"
    # leer bases de datos
    df = pd.read_csv(path, index_col=0)
    names = pd.read_csv(path_names, index_col=0).reset_index(drop=True)
    columns = list(pd.DataFrame(names.iloc[0]).reset_index(drop=True)[0])
    names = names.iloc[1:, :]
    names.columns = columns

    campo_descripcion = "DESCRIPCION"
    if filename == "R_MOVIMIENTOS_CARGAdata":
        campo_descripcion = "DESCIPCION"
    names = names[["CAMPO", campo_descripcion]]
    names.columns = ["CAMPO", "DESCRIPCION"]
    names = drop_spaces_data(names)
    for i in range(len(names)):
        columna = names["CAMPO"].iloc[i]
        descripcion = names["DESCRIPCION"].iloc[i]
        if campo_descripcion == "DESCIPCION":
            columna = columna.upper()
        new_name = columna + ": " + descripcion
        df.rename(columns={columna: new_name}, inplace=True)
    # porcentaje de aceptación de la columna
    porcentaje_aceptacion = 30
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
    print("Las columnas eliminadas son: ", drop_cols)
    df.drop(columns=drop_cols, inplace=True)
    restant_cols = list(df.columns)
    print("Las columnas restantes son: ", restant_cols)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ver el color del html y si se calculan las correlaciones
    dark_mode = True
    minimal_mode = True
    aditional = True

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

    titulo = f"Exploratory Analysis: Empresas Rey-{filename}{title_mode}"
    titulo = titulo.replace(" ", "-").replace("-", " ").title()
    prof = ProfileReport(
        df,
        title=titulo,
        explorative=False,
        minimal=minimal_mode,
        dark_mode=dark_mode)

    # guardar el html
    path_output =\
        f'data/output-data/templates/{filename}-exploratory{type_html}{mode}.html'
    prof.to_file(output_file=path_output)
