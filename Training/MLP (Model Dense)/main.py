import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler # type: ignore
from keras_tuner import RandomSearch  # Keras Tuner pour optimiser les hyperparamètres

# Charger les données
def load_dataset(filepath):
    data = pd.read_csv(filepath, parse_dates=['Date'])
    return data

# Préparer les données pour un modèle Dense
def prepare_dense_data(data):
    # Calcul des indicateurs techniques
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'MACD', 'ADX']
    data_to_use = data[features].dropna().values  # Supprimer les NaN
    
    # Normalisation
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_use)
    
    labels = data['Close'].iloc[len(data) - len(scaled_data):].values  # Utiliser la colonne 'Close' comme label
    return scaled_data, labels, scaler

# Fonction pour ajuster le taux d'apprentissage
def adjust_learning_rate(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 5
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

# Fonction de construction du modèle pour Keras Tuner
def build_dense_model(hp):
    model = Sequential()
    
    # Hyperparamètres : Nombre de neurones et taux de dropout
    dense_units = hp.Choice('dense_units', [64, 128, 256])
    dropout_rate = hp.Choice('dropout_rate', [0.2, 0.3, 0.5])

    model.add(Dense(dense_units, activation='relu', input_shape=(13,)))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(dense_units // 2, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))  # Prédiction du prix de clôture
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Charger les données et préparer pour le modèle
filepath = 'data/safran_preprocessed_data.csv'
data = load_dataset(filepath)
x, y, scaler = prepare_dense_data(data)

# Diviser les données en ensemble d'entraînement et de validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)

# Utilisation de Keras Tuner pour l'optimisation des hyperparamètres
tuner = RandomSearch(
    build_dense_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='dense_tuner_dir',
    project_name='dense_safran_tuning'
)

# Recherche des meilleurs hyperparamètres
tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# Récupérer le meilleur modèle trouvé
best_model = tuner.get_best_models(num_models=1)[0]

# Callbacks : sauvegarde du meilleur modèle, early stopping et ajustement du taux d'apprentissage
checkpoint = ModelCheckpoint('model_mlp_dense_safran.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(adjust_learning_rate)

# Entraîner le modèle final
history = best_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, lr_scheduler],
    verbose=1
)

# Sauvegarder le scaler
np.save('scaler_mlp_dense_safran.npy', scaler)

print("Entraînement terminé et modèle sauvegardé.")
