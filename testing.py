import numpy as np
from tensorflow.keras.models import load_model 

def load_trained_model(model_path='model_lstm_safran.keras', scaler_path='scaler_safran.npy'):
    try:
        model = load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()
        
        # Afficher les attributs du scaler pour inspecter
        print("Attributs du scaler:", dir(scaler))
        
        # Supprimer l'attribut 'feature_names_in_' s'il existe
        if hasattr(scaler, 'feature_names_in_'):
            delattr(scaler, 'feature_names_in_')
            print("Attribut 'feature_names_in_' supprimé.")
        
        return model, scaler
    except FileNotFoundError:
        print(f"Erreur : Le modèle ou scaler est introuvable.")
        return None, None
