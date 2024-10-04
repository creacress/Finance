from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore
import pandas_ta as ta
import yfinance as yf  # Nouvelle bibliothèque pour récupérer les données du marché

app = Flask(__name__)

def get_market_overview():
    ticker = yf.Ticker("SAF.PA")
    data = ticker.history(period="1d")
    
    # Convertir les types numpy en types standards Python
    market_data = {
        "lastClose": float(data['Close'].iloc[-1]),  # Convertir en float
        "openingPrice": float(data['Open'].iloc[-1]),  # Convertir en float
        "daysRange": f"{data['Low'].iloc[-1]} - {data['High'].iloc[-1]}",
        "volume": int(data['Volume'].iloc[-1]),  # Convertir en int
        "averageVolume": int(ticker.info['averageVolume']),
        "marketCap": int(ticker.info['marketCap']),
        "peRatio": float(ticker.info['trailingPE']),
        "bpa": float(ticker.info['trailingEps']),
    }

    return market_data


@app.route('/market-overview', methods=['GET'])
def market_overview():
    try:
        market_data = get_market_overview()
        return jsonify(market_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Charger les données historiques à partir d'un fichier CSV
def load_historical_data(file_path='data/safran_stock_data.csv'):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Calcul des indicateurs techniques
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    bbands = ta.bbands(data['Close'], length=20)
    if bbands is not None:
        data['Upper_BB'], data['Middle_BB'], data['Lower_BB'] = bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']
    
    macd = ta.macd(data['Close'], fast=12, slow=26)
    if macd is not None:
        data['MACD'] = macd['MACD_12_26_9']
    
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
    if adx is not None:
        data['ADX'] = adx['ADX_14']
    
    # Supprimer les lignes avec des valeurs NaN (causées par des périodes insuffisantes pour certains indicateurs)
    data.dropna(inplace=True)
    
    return data

def load_trained_model(model_path, scaler_path):
    try:
        print(f"Tentative de chargement du modèle depuis le chemin : {model_path}")
        model = load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()

        # Vérification si scaler a des attributs indésirables
        if hasattr(scaler, 'feature_names_in_'):
            del scaler.feature_names_in_

        return model, scaler
    except FileNotFoundError as e:
        print(f"Erreur : Le fichier {e.filename} est introuvable.")
        return None, None

def prepare_lstm_data(data, scaler, seq_length=50):
    # Sélectionner les features utilisées lors de l'entraînement (uniquement numériques)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 
                'Upper_BB', 'Middle_BB', 'Lower_BB', 'MACD', 'ADX']

    # Vérifier que les colonnes nécessaires sont présentes
    if not set(features).issubset(data.columns):
        raise ValueError("Certaines colonnes nécessaires pour la prédiction manquent.")
    
    # Vérifier les données avant normalisation
    print("Données avant normalisation:", data[features].head())

    try:
        scaled_data = scaler.fit_transform(data[features].values)  # Applique la normalisation sur les valeurs numériques
    except Exception as e:
        print(f"Erreur lors de la normalisation des données: {e}")
        raise ValueError("Erreur lors de la normalisation des données.")
    # Vérifier les données après la normalisation
    print("Données après normalisation:", scaled_data[:5])

    sequences = []
    
    # Si le nombre de lignes est inférieur à seq_length, ajuster la séquence ou renvoyer une erreur
    if len(scaled_data) < seq_length:
        raise ValueError(f"Pas assez de données pour créer une séquence. Requiert {seq_length}, mais seulement {len(scaled_data)} lignes disponibles.")

    # Générer les séquences
    for i in range(seq_length, len(scaled_data)):
        sequences.append(scaled_data[i-seq_length:i])

    # Vérifier si des séquences ont été créées
    if len(sequences) == 0:
        raise ValueError("Aucune séquence générée à partir des données reçues.")
    
    return np.array(sequences)

# Mise à jour pour inclure à la fois des données historiques et dynamiques
@app.route('/live-data', methods=['GET'])
def live_data():
    # Charger les données historiques
    data = load_historical_data('data/safran_stock_data.csv')

    # Vérifier si les données sont valides
    if data.empty:
        return jsonify({'error': 'Aucune donnée historique disponible.'}), 500

    # Récupérer les données de marché en direct via yfinance
    live_market_data = get_market_overview()

    # Renvoyer les dernières 51 lignes pour s'assurer que nous avons assez de données pour générer des séquences
    latest_data = data.tail(51)

    # Réinitialiser l'index pour inclure 'Date' comme colonne
    latest_data = latest_data.reset_index()

    # Inclure les données en temps réel et les indicateurs calculés dans la réponse JSON
    return jsonify({
        'current_price': live_market_data['lastClose'],
        'openingPrice': live_market_data['openingPrice'],
        'high': live_market_data['daysRange'].split(" - ")[1],
        'low': live_market_data['daysRange'].split(" - ")[0],
        'volume': live_market_data['volume'],
        'candlestick_data': latest_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI',
                                         'Upper_BB', 'Middle_BB', 'Lower_BB', 'MACD', 'ADX']].to_dict(orient='records')
    })

@app.route('/compare', methods=['POST'])
def compare_models():
    data = request.json  # Données reçues du frontend
    live_df = pd.DataFrame(data['live_data'])  # Convertir les données reçues en DataFrame

    # Supprimer la colonne 'Date' si elle est présente
    live_df = live_df.drop(columns=['Date'], errors='ignore')
    
    model_paths = data['model_paths']  # Liste des chemins de modèles envoyés depuis l'interface
    predictions = {}

    for model_path in model_paths:
        # Définir les chemins des scalers pour chaque modèle
        if model_path == 'model_lstm_safran.keras':
            scaler_path = 'scaler_lstm_safran.npy'
        elif model_path == 'model_mlp_dense_safran.keras':
            scaler_path = 'scaler_mlp_dense_safran.npy'
        else:
            predictions[model_path] = 'Modèle non reconnu'
            continue
        
        model, scaler = load_trained_model(model_path=model_path, scaler_path=scaler_path)
        
        if model is None or scaler is None:
            predictions[model_path] = 'Erreur lors du chargement du modèle ou du scaler'
            continue
        
        # Préparer les données pour la prédiction
        try:
            if model_path == 'model_lstm_safran.keras':
                # Préparer les séquences pour le LSTM
                sequences = prepare_lstm_data(live_df, scaler, seq_length=50)
                prediction = model.predict(sequences[-1].reshape(1, sequences.shape[1], sequences.shape[2]))
            else:
                # Préparation pour le MLP Dense (pas de séquences)
                sequences = scaler.transform(live_df.values)
                prediction = model.predict(sequences[-1].reshape(1, sequences.shape[1]))  # Prédiction directe

            # Inverser la normalisation uniquement pour la colonne 'Close'
            empty_data = np.zeros((1, len(live_df.columns)))
            empty_data[0, live_df.columns.get_loc('Close')] = prediction[0, 0]
            predicted_close = scaler.inverse_transform(empty_data)[:, live_df.columns.get_loc('Close')][0]

            predictions[model_path] = predicted_close
        except Exception as e:
            predictions[model_path] = f'Erreur de prédiction : {str(e)}'

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
