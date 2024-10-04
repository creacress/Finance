// Initialisation du graphique avec Chart.js
let stockChart;

// Fonction pour analyser les prédictions et donner un conseil d'investissement
function analyzePrediction(lstmPrediction, mlpPrediction, currentPrice) {
    let trend = '';
    let advice = '';

    if (lstmPrediction > currentPrice && mlpPrediction > currentPrice) {
        trend = 'Haussière';
        advice = 'Acheter';
    } else if (lstmPrediction < currentPrice && mlpPrediction < currentPrice) {
        trend = 'Baissière';
        advice = 'Vendre';
    } else {
        trend = 'Incertaine';
        advice = 'Attendre, prudence recommandée';
    }

    return { trend, advice };
}

// Fonction pour afficher les indicateurs techniques
function displayTechnicalIndicators(rsi, macd, adx) {
    document.getElementById('rsiValue').innerText = rsi ?? 'N/A';
    document.getElementById('macdValue').innerText = macd ?? 'N/A';
    document.getElementById('adxValue').innerText = adx ?? 'N/A';
}

// Fonction pour estimer le risque basé sur la divergence des modèles
function estimateRisk(lstmPrediction, mlpPrediction) {
    const difference = Math.abs(lstmPrediction - mlpPrediction);

    if (difference < 10) {
        return 'Faible';
    } else if (difference < 30) {
        return 'Moyen';
    } else {
        return 'Élevé';
    }
}

// Fonction principale pour récupérer les prédictions et analyser les résultats
async function fetchPrediction() {
    try {
        const liveData = await fetch('http://127.0.0.1:5000/live-data');
        const liveDataJson = await liveData.json();

        const candlestickData = liveDataJson.candlestick_data;
        const currentPrice = liveDataJson.current_price;

        if (!candlestickData || candlestickData.length === 0) {
            throw new Error('Les données de chandelier sont manquantes.');
        }

        const selectedModels = ['model_lstm_safran.keras', 'model_mlp_dense_safran.keras'];
        const response = await fetch('http://127.0.0.1:5000/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ live_data: candlestickData, model_paths: selectedModels })
        });

        const data = await response.json();
        const lstmPred = parseFloat(data['model_lstm_safran.keras']).toFixed(2);
        const mlpPred = parseFloat(data['model_mlp_dense_safran.keras']).toFixed(2);

        // Mise à jour des valeurs de prédictions
        document.getElementById('lstmPredictedPrice').innerText = lstmPred;
        document.getElementById('mlpPredictedPrice').innerText = mlpPred;
        document.getElementById('currentPrice').innerText = currentPrice;

        // Analyse des prédictions pour les conseils d'investissement
        const { trend, advice } = analyzePrediction(lstmPred, mlpPred, currentPrice);
        document.getElementById('marketTrend').innerText = trend;
        document.getElementById('investmentAdvice').innerText = advice;

        // Affichage du risque
        const riskLevel = estimateRisk(lstmPred, mlpPred);
        document.getElementById('riskLevel').innerText = riskLevel;

        // Afficher les indicateurs techniques
        displayTechnicalIndicators(liveDataJson.RSI, liveDataJson.MACD, liveDataJson.ADX);

        // Mettre à jour le graphique
        updateChart(candlestickData);

    } catch (error) {
        console.error("Erreur lors de la récupération de la prédiction:", error);
    }
}

// Initialisation du graphique avec Chart.js
function initializeChart() {
    const ctx = document.getElementById('stockChart').getContext('2d');

    stockChart = new Chart(ctx, {
        type: 'candlestick',  // Graphique en chandeliers
        data: {
            labels: [],  // Les dates
            datasets: [{
                label: 'Prix de l\'action',
                data: [],  // Les données de chandeliers (ou les prix)
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
                    type: 'time',
                    time: {
                        unit: 'day',  // Adapter l'unité de temps (jour, minute, etc.)
                        tooltipFormat: 'dd MMM yyyy'  // Format des dates dans le tooltip
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Prix (EUR)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(tooltipItem) {
                            return `Prix: ${tooltipItem.raw.c} EUR`;  // Afficher le prix dans le tooltip
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutBounce'
            }
        }
    });
}

// Fonction pour mettre à jour le graphique avec les nouvelles données
function updateChart(candlestickData) {
    if (!candlestickData || candlestickData.length === 0) {
        console.error("Les données du graphique sont vides ou non valides.");
        return;
    }

    const labels = candlestickData.map(item => new Date(item.Date));
    const prices = candlestickData.map(item => ({
        o: item.Open,  // Open
        h: item.High,  // High
        l: item.Low,   // Low
        c: item.Close  // Close
    }));

    console.log("Données des prix :", prices); // Debug : voir les données reçues

    // Mettre à jour les données du graphique
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
    
    // Initialiser le graphique lors du chargement de la page
    initializeChart();

    fetchMarketData();  // Obtenir les données du marché au chargement de la page

    setInterval(fetchMarketData, 5 * 60 * 1000);  // Mise à jour des données toutes les 5 minutes
});
