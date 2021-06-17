from pandas_profiling import ProfileReport


def profiling_report(df, minimal_mode=False, dark_mode=True):
    """
    Utiliza la libreria pandas_profiling para hacer una exploración visual
    rápida de los datos
    Parameters
    ----------
    df : dataframe
        dataframe with data to analyse.
    minimal_mode : string, optional
        En el caso de que sea True, hace cálculo de correlaciones no lineales.
        The default is False.
    dark_mode : string, optional
        si es en el modo oscuro o no. The default is True.
    Returns
    -------
    .html con la exploración de los datos.
    """

    # esto hace la logica de como guardar el archivo nomás
    if dark_mode:
        type_html = "-black"
    else:
        type_html = ""
    if minimal_mode:
        title_mode = "no expensive computations"
        mode = title_mode.replace(" ", "-")
    else:
        title_mode = ""
        mode = title_mode.replace(" ", "-")

    title = "Exploratory Data Analysis: Floating Data"
    prof = ProfileReport(df,
                         title=title,
                         explorative=False,
                         minimal=minimal_mode,
                         orange_mode=dark_mode)
    # guardar el html
    path_output =\
        f'results/exploratory-analysis/{mode}-eda.html'
    prof.to_file(output_file=path_output)
