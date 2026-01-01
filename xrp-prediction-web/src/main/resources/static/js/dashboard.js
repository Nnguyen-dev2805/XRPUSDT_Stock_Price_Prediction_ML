// Initialize Chart when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializePriceChart();
    updateDashboardStats();
});

// Chart instance
let chartInstance = null;

function initializePriceChart() {
    const chartCanvas = document.getElementById('priceChart');
    if (!chartCanvas) return;

    fetch('/api/chart-data')
        .then(response => response.json())
        .then(data => {
            if (data.length === 0) {
                console.log('No chart data available');
                return;
            }

            const labels = data.map(d => d.date);
            const actualPrices = data.map(d => d.actual);
            const pred1d = data.map(d => d.pred1d);
            const pred3d = data.map(d => d.pred3d);
            const pred5d = data.map(d => d.pred5d);
            const pred7d = data.map(d => d.pred7d);

            const ctx = chartCanvas.getContext('2d');
            
            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: actualPrices,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.05)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 4,
                            pointBackgroundColor: '#3498db',
                            pointBorderColor: 'white',
                            pointBorderWidth: 2,
                            pointHoverRadius: 6,
                        },
                        {
                            label: 'Predicted (1D)',
                            data: pred1d,
                            borderColor: '#e74c3c',
                            borderDash: [5, 5],
                            borderWidth: 2,
                            fill: false,
                            tension: 0.4,
                            pointRadius: 3,
                            pointBackgroundColor: '#e74c3c',
                            pointBorderColor: 'white',
                            pointBorderWidth: 1,
                        },
                        {
                            label: 'Predicted (3D)',
                            data: pred3d,
                            borderColor: '#2ecc71',
                            borderDash: [5, 5],
                            borderWidth: 2,
                            fill: false,
                            tension: 0.4,
                            pointRadius: 3,
                            pointBackgroundColor: '#2ecc71',
                            pointBorderColor: 'white',
                            pointBorderWidth: 1,
                        },
                        {
                            label: 'Predicted (5D)',
                            data: pred5d,
                            borderColor: '#f39c12',
                            borderDash: [5, 5],
                            borderWidth: 2,
                            fill: false,
                            tension: 0.4,
                            pointRadius: 3,
                            pointBackgroundColor: '#f39c12',
                            pointBorderColor: 'white',
                            pointBorderWidth: 1,
                        },
                        {
                            label: 'Predicted (7D)',
                            data: pred7d,
                            borderColor: '#9b59b6',
                            borderDash: [5, 5],
                            borderWidth: 2,
                            fill: false,
                            tension: 0.4,
                            pointRadius: 3,
                            pointBackgroundColor: '#9b59b6',
                            pointBorderColor: 'white',
                            pointBorderWidth: 1,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                font: {
                                    size: 12,
                                    weight: '500'
                                },
                                padding: 20,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleFont: { size: 14, weight: 'bold' },
                            bodyFont: { size: 12 },
                            padding: 12,
                            cornerRadius: 8,
                            displayColors: true,
                            borderColor: '#ddd',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                font: { size: 11 },
                                callback: function(value) {
                                    return value.toFixed(4);
                                }
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)',
                                drawBorder: false
                            }
                        },
                        x: {
                            ticks: {
                                font: { size: 11 }
                            },
                            grid: {
                                display: false,
                                drawBorder: false
                            }
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading chart data:', error));
}

function updateDashboardStats() {
    fetch('/api/dashboard-stats')
        .then(response => response.json())
        .then(data => {
            console.log('Dashboard stats:', data);
            // Stats will be rendered server-side via Thymeleaf
            // This function can be used for dynamic updates in the future
        })
        .catch(error => console.error('Error loading stats:', error));
}

// Auto-refresh every 5 minutes
setInterval(function() {
    initializePriceChart();
    updateDashboardStats();
}, 5 * 60 * 1000);
