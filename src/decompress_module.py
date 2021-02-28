import pandas as pd


def decompress_data(filename="data/raw-data/data-empresas-rey.xlsx"):
    # descomprimir todas las hojas
    data = pd.ExcelFile(filename, engine="openpyxl")
    sheet_names = data.sheet_names
    for file in sheet_names:
        print("Descomprimiendo: ", file, "...")
        df = data.parse(sheet_name=file)
        path = f"data/input-data/{file}.csv"
        df.to_csv(path)
