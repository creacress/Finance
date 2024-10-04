# Prédiction du Marché Boursier de Safran

Ce projet est une application **Electron.js** qui permet de prédire les prix des actions de Safran (SAF.PA) à l'aide de deux modèles d'intelligence artificielle : **LSTM** et **MLP Dense**. Le backend est alimenté par Flask, et les données sont récupérées en temps réel depuis Yahoo Finance via l'API `yfinance`. L'application affiche les données du marché en direct et compare les prédictions des deux modèles.

## Fonctionnalités

- **Données du marché en temps réel** : Récupération des informations du marché via Yahoo Finance, telles que le prix actuel, le volume, et la plage de prix de la journée.
- **Prédiction des prix des actions** : Utilisation de deux modèles IA (LSTM et MLP Dense) pour prédire les prix futurs des actions.
- **Comparaison des modèles IA** : Affichage des prédictions issues des deux modèles pour permettre une **comparaison directe** de leurs performances.
- **Graphique en Chandeliers Japonais** : Affichage d’un graphique en temps réel avec les données historiques, mis à jour régulièrement.

## Technologies Utilisées

- **Electron.js** : Pour le développement de l'application desktop.
- **Flask** : API backend pour servir les données du marché et les prédictions.
- **TensorFlow/Keras** : Pour l'entraînement et l'exécution des modèles IA.
- **Chart.js** : Pour le rendu du graphique des chandeliers.
- **yfinance** : Pour récupérer les données du marché en temps réel.
- **pandas_ta** : Pour le calcul des indicateurs techniques tels que MACD, RSI, et ADX.

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/votrenomutilisateur/safran-stock-prediction.git
   cd safran-stock-prediction
   ```

2. **Installer les dépendances Python** :
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # Sur Windows, utilisez `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Installer les dépendances Node.js** :
   ```bash
   cd ../frontend
   npm install
   ```

4. **Lancer le serveur Flask** :
   ```bash
   cd backend
   flask run
   ```

5. **Lancer l'application Electron** :
   ```bash
   cd ../frontend
   npm start
   ```

## Comparaison des Modèles IA

### Modèle LSTM
Le modèle LSTM est entraîné sur des données historiques d’actions, intégrant des indicateurs techniques tels que :
- **SMA** (Moyenne Mobile Simple) à 50 et 200 jours
- **RSI** (Indice de Force Relative)
- **Bollinger Bands**
- **MACD** (Moyenne Mobile Convergente Divergente)
- **ADX** (Indice Directionnel Moyen)

### Modèle MLP Dense
Le modèle MLP Dense, un réseau de neurones à couches entièrement connectées, utilise les mêmes indicateurs techniques pour prédire les prix des actions.

### Comparaison des Prédictions
L'application permet de **comparer directement** les prédictions des deux modèles. Une section dédiée affiche les résultats des modèles **LSTM** et **MLP Dense** ainsi que la **différence** entre leurs prédictions, offrant ainsi une meilleure visibilité sur les performances respectives des deux approches.

## Utilisation

1. **Lancer le Serveur Flask** : Assurez-vous que le serveur Flask est en marche à l'adresse `http://127.0.0.1:5000`.
2. **Ouvrir l'Application Electron** : Démarrez l'application Electron pour voir les données du marché en temps réel, les graphiques en chandeliers, et les prix prédits.
3. **Comparer les Prédictions** : Cliquez sur le bouton 'Predict' pour comparer les prédictions des deux modèles IA. Les résultats seront affichés et la différence entre les deux prédictions sera calculée.
## Captures d'Écran
![projet_electron](https://github.com/user-attachments/assets/0a8c729d-3e02-466b-ab51-97c0b7ce63dd)
## Licence

Ce projet est sous licence MIT.
