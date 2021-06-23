import pandas as pd


def initial_shift_time(date, turno):
    """
    Ver en funcion de la fecha la empezada y finalizada del turno
    Parameters
    ----------
    date : datetime
        DESCRIPTION.
    turno : TYPE
        DESCRIPTION.
    Returns
    -------
    date : TYPE
        DESCRIPTION.
    """
    if turno == "Noche":
        date = date.replace(hour=20, minute=0, second=0)
    else:
        date = date.replace(hour=8, minute=0, second=0)
    return date


def get_columns_correlated(df, method="pearson", threshold=0.8):
    """
    Obtener las columnas altamente relacionadas del ads final, en función
    de distintos thresholds
    Parameters
    ----------
    df : dataframe
        dataframe sobre el cual sacar correlaciones.
    method : string, optional
        metodo de correlacion. The default is "pearson".
    threshold : float, optional
        umbral sobre el cual se considera una correlacion positivo o negagita.
        The default is 0.8.
    Returns
    -------
    correlated_cols : dataframe
        DESCRIPTION.
    """
    corr_mt = df.corr(method=method)
    corr_mt.reset_index(drop=False, inplace=True)
    corr_mt.rename(columns={"index": "columnas"}, inplace=True)
    correlated_cols = []
    columns = list(corr_mt.columns)
    columns.remove("columnas")
    for col in columns:
        for i in range(len(corr_mt)):
            index = corr_mt["columnas"].iloc[i]
            # print(col, index)
            if index == col:
                pass
            else:
                value = corr_mt[col].iloc[i]
                if value > threshold:
                    string = col + index
                    string = ''.join(sorted(string))
                    corr_cols = [col, index, value,
                                 "correlacion postiviva", string]
                    correlated_cols.append(corr_cols)
                if value < -threshold:
                    string = col + index
                    string = ''.join(sorted(string))
                    corr_cols = [col, index, value,
                                 "correlacion negativa", string]
                    correlated_cols.append(corr_cols)

    correlated_cols = pd.DataFrame(correlated_cols,
                                   columns=["col1", "col2", "correlacion",
                                            "descripcion", "id"])
    correlated_cols.drop_duplicates(subset=["id"], inplace=True)
    correlated_cols.drop(columns=["id"], inplace=True)
    correlated_cols.sort_values(by=["correlacion"], ascending=False,
                                inplace=True)
    # no me interesan variables obvias
    correlated_cols = correlated_cols[correlated_cols["correlacion"] != 1]
    correlated_cols.reset_index(drop=True, inplace=True)
    correlated_cols["metodo"] = method
    return correlated_cols
