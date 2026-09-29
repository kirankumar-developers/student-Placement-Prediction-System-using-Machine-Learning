"""
Student Placement Prediction System - Flask Web Application
Provides full-stack interactive dashboard, dynamic prediction forms,
multi-algorithm benchmarking, and REST API endpoints.
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

from prediction import get_predictor, PlacementPredictor
import train_model

app = Flask(__name__)
app.secret_key = 'student_placement_secret_key_2026'

# Ensure required models and artifacts exist on startup
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

def ensure_models_trained():
    """Checks if models and metrics exist; if not, trains them automatically."""
    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    pipeline_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')
    metrics_path = os.path.join(OUTPUTS_DIR, 'evaluation_metrics.json')

    if not os.path.exists(meta_path) or not os.path.exists(pipeline_path) or not os.path.exists(metrics_path):
        print("[STARTUP] Models or evaluation artifacts not found. Initiating full training pipeline...")
        train_model.train_and_evaluate_all()
    else:
        print("[STARTUP] Pre-trained models and artifacts detected. System ready.")

ensure_models_trained()


def load_metrics_and_metadata():
    """Helper to retrieve cached evaluation metrics and schema metadata."""
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, 'evaluation_metrics.json'), 'r') as f:
        metrics = json.load(f)
    comparison_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'model_comparison.csv'))
    return metadata, metrics, comparison_df


@app.route('/')
def index():
    """Home Dashboard displaying dataset KPIs, Champion Model, and Feature Overview."""
    try:
        metadata, metrics, comparison_df = load_metrics_and_metadata()
        total_students = metadata.get('total_samples', 1000)
        placed_count = metadata.get('placed_count', 737)
        not_placed_count = metadata.get('not_placed_count', 263)
        placement_rate = round((placed_count / total_students) * 100, 1) if total_students else 0

        models_summary = comparison_df.to_dict(orient='records')

        return render_template(
            'index.html',
            total_students=total_students,
            placed_count=placed_count,
            not_placed_count=not_placed_count,
            placement_rate=placement_rate,
            features=metadata.get('features', {}),
            target_col=metadata.get('target_col', 'Placement'),
            metrics=metrics,
            models_summary=models_summary
        )
    except Exception as e:
        flash(f"Error loading dashboard data: {str(e)}", "error")
        return render_template('500.html'), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Dynamic Student Placement Prediction Form (GET) and Inference Handler (POST)."""
    predictor = get_predictor()
    metadata, metrics, _ = load_metrics_and_metadata()
    features = metadata.get('features', {})
    available_models = predictor.get_available_models()

    if request.method == 'POST':
        try:
            selected_model = request.form.get('selected_model', 'best_model')
            
            # Extract dynamically according to actual feature names
            raw_input = {}
            for feat_name in predictor.feature_names:
                raw_input[feat_name] = request.form.get(feat_name)

            # Perform prediction
            prediction_response = predictor.predict_single(raw_input, model_name=selected_model)

            if not prediction_response.get('success', False):
                errors = prediction_response.get('errors', ['Validation failed.'])
                for err in errors:
                    flash(err, 'error')
                return render_template(
                    'predict.html',
                    features=features,
                    available_models=available_models,
                    form_data=raw_input
                )

            return render_template('result.html', result=prediction_response)

        except Exception as e:
            flash(f"Prediction failed: {str(e)}", 'error')
            return redirect(url_for('predict'))

    return render_template(
        'predict.html',
        features=features,
        available_models=available_models,
        form_data={}
    )


@app.route('/comparison')
def comparison():
    """Comprehensive Model Benchmark Hub comparing all 5 algorithms."""
    try:
        metadata, metrics, comparison_df = load_metrics_and_metadata()
        models_summary = comparison_df.to_dict(orient='records')
        best_model_name = metrics.get('best_model')
        best_model_row = comparison_df[comparison_df['Algorithm'] == best_model_name].iloc[0].to_dict()

        return render_template(
            'comparison.html',
            metrics=metrics,
            models_summary=models_summary,
            models_summary_json=json.dumps(models_summary),
            best_model_row=best_model_row
        )
    except Exception as e:
        flash(f"Error loading comparison benchmark: {str(e)}", 'error')
        return render_template('500.html'), 500


@app.route('/visualizations')
def visualizations():
    """High-resolution EDA and Performance Visualizations Gallery."""
    return render_template('visualizations.html')


# ===================== REST API ENDPOINTS =====================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API for programmatic placement prediction.
    Accepts JSON body with student feature key-values and optional 'model' name.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'JSON payload is required.'}), 400

        model_name = data.get('model', 'best_model')
        predictor = get_predictor()

        # Extract features dictionary
        student_features = data.get('features', data)
        result = predictor.predict_single(student_features, model_name=model_name)

        if not result.get('success'):
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """REST API returning latest model evaluation metrics and comparison table."""
    try:
        _, metrics, comparison_df = load_metrics_and_metadata()
        return jsonify({
            'success': True,
            'metrics': metrics,
            'table': comparison_df.to_dict(orient='records')
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ===================== ERROR HANDLERS =====================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("==========================================================")
    print("  STUDENT PLACEMENT PREDICTION SYSTEM - FLASK SERVER      ")
    print(f"  Server running on port: {port}                         ")
    print("==========================================================")
    app.run(host='0.0.0.0', port=port, debug=False)

