import yfinance as yf
import pandas as pd


# Télécharger les données pour Safran (SAF.PA)
def download_stock_data(symbol, start_date, end_date, filename):
    # Télécharger les données
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Sauvegarder les données dans un fichier CSV
    stock_data.to_csv(filename)
    print(f"Les données ont été téléchargées et enregistrées sous {filename}")

# Paramètres
symbol = 'SAF.PA'  # Symbole boursier pour Safran sur Yahoo Finance
start_date = '2000-01-03'
end_date = '2024-09-28'
filename = 'safran_stock_data.csv'

# Appel de la fonction pour télécharger et sauvegarder
download_stock_data(symbol, start_date, end_date, filename)