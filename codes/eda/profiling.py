from src.exploratory_pandas_profiling import exploratory_analysis


filename = "MKPFdata"
path = f"data/input-data/{filename}.csv"

# hacer el exploratorio para que lo vaya a dejar a la carpeta de ouptut data
exploratory_analysis(path, filename, nombre="Empresas Rey",
                     porcentaje_aceptacion=30,
                     dark_mode=True,
                     minimal_mode=True,
                     aditional=True)

# con calculo de correlaciones
exploratory_analysis(path, filename, nombre="Empresas Rey",
                     porcentaje_aceptacion=30,
                     dark_mode=True,
                     minimal_mode=False,
                     aditional=True)
