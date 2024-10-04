// Fonction pour récupérer les données du marché en direct depuis Flask
async function fetchMarketData() {
    try {
        const response = await fetch('http://127.0.0.1:5000/live-data');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            console.log("Données JSON parsées:", data);
            
            // Utilisation des données du marché en direct
            const currentPriceElem = document.getElementById('currentPrice');
            if (currentPriceElem) currentPriceElem.innerText = data.current_price ?? 'N/A';

            const openingPriceElem = document.getElementById('openingPrice');
            const highPriceElem = document.getElementById('highPrice');
            const lowPriceElem = document.getElementById('lowPrice');
            const volumeElem = document.getElementById('volume');

            // Affichage des données dynamiques récupérées depuis l'API
            if (openingPriceElem) openingPriceElem.innerText = data.openingPrice ?? 'N/A';
            if (highPriceElem) highPriceElem.innerText = data.high ?? 'N/A';
            if (lowPriceElem) lowPriceElem.innerText = data.low ?? 'N/A';
            if (volumeElem) volumeElem.innerText = data.volume ?? 'N/A';

            // Mettre à jour le graphique avec les données candlestick
            if (Array.isArray(data.candlestick_data) && data.candlestick_data.length > 0) {
                updateChart(data.candlestick_data);
            } else {
                console.error('candlestick_data est vide ou n\'est pas un tableau.');
            }
        } else {
            throw new Error('Données reçues ne sont pas au format JSON.');
        }
    } catch (error) {
        console.error("Erreur lors de la récupération des données du marché:", error);
    }
}

// Fonction pour envoyer les données au backend Flask pour la prédiction avec les deux modèles
async function fetchPrediction() {
    try {
        const liveData = await fetch('http://127.0.0.1:5000/live-data');
        const liveDataJson = await liveData.json();

        const candlestickData = liveDataJson.candlestick_data;

        if (!candlestickData || candlestickData.length === 0) {
            throw new Error('Les données de chandelier sont manquantes.');
        }

        // Envoi des données pour les deux modèles
        const selectedModels = ['model_lstm_safran.keras', 'model_mlp_dense_safran.keras'];

        // Afficher "Loading..." avant la réception des résultats
        document.getElementById('lstmResult').innerText = 'Loading...';
        document.getElementById('mlpResult').innerText = 'Loading...';

        const response = await fetch('http://127.0.0.1:5000/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                live_data: candlestickData,
                model_paths: selectedModels 
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Arrondir les résultats à 2 décimales pour un affichage plus propre
        const lstmPred = parseFloat(data['model_lstm_safran.keras']).toFixed(2);
        const mlpPred = parseFloat(data['model_mlp_dense_safran.keras']).toFixed(2);

        // Afficher les résultats arrondis dans la section prédiction
        document.getElementById('lstmPredictedPrice').innerText = lstmPred ?? 'N/A';
        document.getElementById('mlpPredictedPrice').innerText = mlpPred ?? 'N/A';

        // Mettre à jour la section "Results Comparison" avec les résultats
        document.getElementById('lstmResult').innerText = lstmPred;
        document.getElementById('mlpResult').innerText = mlpPred;

        // Calculer et afficher la différence entre les modèles
        const difference = Math.abs(lstmPred - mlpPred).toFixed(2);
        document.getElementById('modelDifference').innerText = difference;
        
    } catch (error) {
        console.error("Erreur lors de la récupération de la prédiction:", error);
    }
}

// Initialisation du graphique avec Chart.js
const ctx = document.getElementById('stockChart').getContext('2d');
const stockChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [], // Les dates/temps
        datasets: [{
            label: 'Prix de l\'action',
            data: [], // Les prix des actions
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 2,
            pointBackgroundColor: '#1e90ff',
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            fill: true
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                type: 'time',  // Pour l'adaptation temporelle
                time: {
                    unit: 'day',  // Adapter selon l'intervalle (jour ou minute selon les données)
                    tooltipFormat: 'dd MMM yyyy'
                }
            },
            y: {
                beginAtZero: false
            }
        },
        animation: {
            duration: 1000,
            easing: 'easeInOutBounce'
        }
    }
});

// Fonction pour mettre à jour le graphique avec les nouvelles données
function updateChart(candlestickData) {
    const labels = candlestickData.map(item => new Date(item.Date));
    const prices = candlestickData.map(item => item.Close);

    stockChart.data.labels = labels;
    stockChart.data.datasets[0].data = prices;
    stockChart.update();
}

// Gestion de l'événement 'DOMContentLoaded'
document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predictBtn');
    
    if (predictBtn) {
        predictBtn.addEventListener('click', () => {
            fetchPrediction();  // Appel de la fonction pour prédire les prix
        });
    }
    
    fetchMarketData(); // Obtenir les données du marché

    setInterval(fetchMarketData, 5 * 60 * 1000);  // Mise à jour des données toutes les 5 minutes
});
