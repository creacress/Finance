import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
import keras_tuner as kt  # Importer keras-tuner pour l'optimisation des hyperparamètres
import ta  # Bibliothèque pour calculer les indicateurs techniques

# Charger les données depuis ton dataset
def load_dataset(filepath):
    try:
        data = pd.read_csv(filepath, parse_dates=['Date'])
        return data
    except FileNotFoundError:
        print(f"Erreur : Le fichier {filepath} est introuvable.")
        return None

# Préparer les données avec des indicateurs supplémentaires (MACD, ADX)
def prepare_lstm_data(data, seq_length=50):
    # Calculer MACD
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculer ADX avec pandas_ta
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)
    
    # On utilise toutes les colonnes sauf 'Date' pour l'entraînement
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'MACD', 'ADX']
    data_to_use = data[features].values
    
    # Normalisation des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_use)
    
    sequences, labels = [], []
    for i in range(seq_length, len(scaled_data)):
        sequences.append(scaled_data[i-seq_length:i])
        labels.append(scaled_data[i, 3])  # On prend le prix de clôture comme cible
    
    return np.array(sequences), np.array(labels), scaler

# Fonction de construction du modèle pour keras-tuner
def build_lstm_model(hp):
    model = Sequential()
    
    # Couche CNN avant LSTM
    model.add(Conv1D(filters=64, kernel_size=hp.Choice('kernel_size', [2, 3, 5]), activation='relu', input_shape=(50, 13)))
    model.add(MaxPooling1D(pool_size=hp.Choice('pool_size', [2, 3])))
    
    # LSTM avec BatchNormalization
    lstm_units = hp.Choice('lstm_units', [64, 128, 256])
    dropout_rate = hp.Choice('dropout_rate', [0.2, 0.3, 0.5])
    
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(units=lstm_units, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Couche Dense avec L2 Regularization
    model.add(Dense(units=50, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fonction pour ajuster le taux d'apprentissage plus fréquemment
def adjust_learning_rate(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 5  # Réduction tous les 5 epochs pour affiner l'apprentissage
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

# Entraîner et sauvegarder le modèle avec keras-tuner
def train_and_save_model(filepath, seq_length=50, model_path='model_lstm_safran.keras'):
    data = load_dataset(filepath)
    if data is None:
        return None, None
    
    # Préparer les données pour LSTM
    sequences, labels, scaler = prepare_lstm_data(data, seq_length)
    
    # Diviser les données en training et validation
    split_idx = int(len(sequences) * 0.8)
    x_train, x_val = sequences[:split_idx], sequences[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    
    # Utilisation de keras-tuner pour l'optimisation des hyperparamètres
    tuner = kt.RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='lstm_safran_tuning'
    )
    
    # Recherche des meilleurs hyperparamètres
    tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))  # Moins d'epochs pour la recherche
    
    # Récupérer le meilleur modèle
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Callbacks : sauvegarde du meilleur modèle, early stopping et ajustement du taux d'apprentissage
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(adjust_learning_rate)
    
    # Entraînement du modèle final
    history = best_model.fit(
        x_train, y_train, 
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint, early_stopping, lr_scheduler],
        verbose=1
    )
    
    # Sauvegarder le scaler pour usage futur (inversion des prédictions)
    np.save('scaler_safran.npy', scaler)
    print(f"Modèle entraîné et sauvegardé sous {model_path}")
    
    return best_model, scaler

# Utilisation
filepath = 'data/safran_preprocessed_data.csv'
model, scaler = train_and_save_model(filepath)
