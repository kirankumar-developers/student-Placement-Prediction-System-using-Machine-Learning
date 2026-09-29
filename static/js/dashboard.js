/**
 * Student Placement Prediction System - Dashboard & Interactivity
 */

document.addEventListener('DOMContentLoaded', function () {
    // Initialize Tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Preset Student Profiles Handler
    setupPresetProfiles();

    // Init charts if containers exist
    initDashboardCharts();
    initComparisonCharts();
});

/**
 * Quick autofill of student profiles on prediction form
 */
function setupPresetProfiles() {
    const presets = {
        'high': {
            'CGPA': 9.2,
            '10th_Percentage': 88.5,
            '12th_Percentage': 85.0,
            'Internships': 2,
            'Aptitude_Score': 85.0,
            'Communication_Score': 9.0,
            'Technical_Skill_Score': 9.0,
            'Projects': 4,
            'Certifications': 3,
            'Backlogs': 0
        },
        'moderate': {
            'CGPA': 7.6,
            '10th_Percentage': 73.0,
            '12th_Percentage': 70.0,
            'Internships': 1,
            'Aptitude_Score': 68.0,
            'Communication_Score': 7.2,
            'Technical_Skill_Score': 7.0,
            'Projects': 2,
            'Certifications': 2,
            'Backlogs': 0
        },
        'low': {
            'CGPA': 5.8,
            '10th_Percentage': 56.0,
            '12th_Percentage': 52.0,
            'Internships': 0,
            'Aptitude_Score': 45.0,
            'Communication_Score': 5.5,
            'Technical_Skill_Score': 5.0,
            'Projects': 1,
            'Certifications': 0,
            'Backlogs': 2
        }
    };

    document.querySelectorAll('[data-preset]').forEach(button => {
        button.addEventListener('click', function () {
            const presetKey = this.getAttribute('data-preset');
            const data = presets[presetKey];
            if (!data) return;

            for (const [key, value] of Object.entries(data)) {
                const input = document.getElementById(`field_${key}`);
                if (input) {
                    input.value = value;
                    input.classList.add('is-valid');
                    setTimeout(() => input.classList.remove('is-valid'), 1200);
                }
            }
        });
    });
}

/**
 * Home Dashboard Interactive Charts
 */
function initDashboardCharts() {
    const ctxOverview = document.getElementById('chartOverviewDist');
    if (ctxOverview && window.dashboardData) {
        new Chart(ctxOverview, {
            type: 'doughnut',
            data: {
                labels: ['Placed', 'Not Placed'],
                datasets: [{
                    data: [window.dashboardData.placed_count, window.dashboardData.not_placed_count],
                    backgroundColor: ['#10B981', '#EF4444'],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { font: { family: 'Plus Jakarta Sans', weight: 600 } }
                    }
                },
                cutout: '70%'
            }
        });
    }
}

/**
 * Model Comparison Interactive Multi-Metric Chart
 */
function initComparisonCharts() {
    const ctxComp = document.getElementById('chartModelComparison');
    if (ctxComp && window.modelComparisonData) {
        const algorithms = window.modelComparisonData.map(d => d.Algorithm);
        const accuracy = window.modelComparisonData.map(d => d.Accuracy * 100);
        const precision = window.modelComparisonData.map(d => d.Precision * 100);
        const recall = window.modelComparisonData.map(d => d.Recall * 100);
        const f1 = window.modelComparisonData.map(d => d['F1 Score'] * 100);

        new Chart(ctxComp, {
            type: 'bar',
            data: {
                labels: algorithms,
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: accuracy,
                        backgroundColor: '#4361EE'
                    },
                    {
                        label: 'Precision (%)',
                        data: precision,
                        backgroundColor: '#3A0CA3'
                    },
                    {
                        label: 'Recall (%)',
                        data: recall,
                        backgroundColor: '#7209B7'
                    },
                    {
                        label: 'F1 Score (%)',
                        data: f1,
                        backgroundColor: '#F72585'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) { return value + "%"; }
                        },
                        grid: { color: '#f1f5f9' }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { font: { family: 'Plus Jakarta Sans', weight: 600 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // Radar Chart for Best Model vs Average Benchmark
    const ctxRadar = document.getElementById('chartModelRadar');
    if (ctxRadar && window.modelComparisonData) {
        const best = window.modelComparisonData[0];
        const avgAcc = window.modelComparisonData.reduce((a, b) => a + b.Accuracy, 0) / window.modelComparisonData.length * 100;
        const avgPrec = window.modelComparisonData.reduce((a, b) => a + b.Precision, 0) / window.modelComparisonData.length * 100;
        const avgRec = window.modelComparisonData.reduce((a, b) => a + b.Recall, 0) / window.modelComparisonData.length * 100;
        const avgF1 = window.modelComparisonData.reduce((a, b) => a + b['F1 Score'], 0) / window.modelComparisonData.length * 100;
        const avgAuc = window.modelComparisonData.reduce((a, b) => a + b['ROC AUC'], 0) / window.modelComparisonData.length * 100;

        new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                datasets: [
                    {
                        label: `${best.Algorithm} (Best Model)`,
                        data: [best.Accuracy * 100, best.Precision * 100, best.Recall * 100, best['F1 Score'] * 100, best['ROC AUC'] * 100],
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        pointBackgroundColor: '#10B981'
                    },
                    {
                        label: 'Average across 5 Models',
                        data: [avgAcc, avgPrec, avgRec, avgF1, avgAuc],
                        borderColor: '#64748B',
                        backgroundColor: 'rgba(100, 116, 139, 0.15)',
                        pointBackgroundColor: '#64748B'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        min: 70,
                        max: 100,
                        ticks: { stepSize: 10 }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { font: { family: 'Plus Jakarta Sans', weight: 600 } }
                    }
                }
            }
        });
    }
}
