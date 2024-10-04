import pandas as pd
from finta import TA
from sklearn.preprocessing import MinMaxScaler

# Lire les données depuis le fichier CSV
def load_data(filename):
    data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    return data

# Ajouter des indicateurs techniques avec finta
def add_technical_indicators(data):
    # Moyennes mobiles simples (SMA) sur 50 et 200 jours
    data['SMA_50'] = TA.SMA(data, 50)
    data['SMA_200'] = TA.SMA(data, 200)
    
    # Relative Strength Index (RSI)
    data['RSI'] = TA.RSI(data)
    
    # Bandes de Bollinger
    bb = TA.BBANDS(data)
    data['Upper_BB'] = bb['BB_UPPER']
    data['Middle_BB'] = bb['BB_MIDDLE']
    data['Lower_BB'] = bb['BB_LOWER']
    
    return data

# Gérer les valeurs manquantes
def handle_missing_values(data):
    # Supprimer les lignes avec des valeurs manquantes
    data = data.dropna()
    return data

# Normaliser les données pour l'entraînement des modèles IA
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, index=data.index, columns=data.columns), scaler

# Prétraitement complet
def preprocess_data(filename):
    # Charger les données
    data = load_data(filename)
    
    # Ajouter des indicateurs techniques
    data = add_technical_indicators(data)
    
    # Gérer les valeurs manquantes
    data = handle_missing_values(data)
    
    # Normaliser les colonnes Open, High, Low, Close, Volume et les indicateurs
    data_to_scale = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Middle_BB', 'Lower_BB']]
    scaled_data, scaler = normalize_data(data_to_scale)
    
    return scaled_data, scaler

# Exécution du prétraitement
filename = 'safran_stock_data.csv'
preprocessed_data, scaler = preprocess_data(filename)

# Sauvegarder les données prétraitées
preprocessed_data.to_csv('safran_preprocessed_data.csv')
print("Les données prétraitées ont été sauvegardées dans 'safran_preprocessed_data.csv'")
