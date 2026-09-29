"""
Prediction Service Module for Student Placement Prediction System
Provides inference pipeline, dynamic feature validation, probability estimation,
and student insight generation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = 'models'

class PlacementPredictor:
    """Encapsulates model inference, data validation, and explanation logic."""
    
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.metadata = None
        self.feature_names = []
        self._load_resources()

    def _load_resources(self):
        """Loads fitted preprocessing pipeline, metadata, and all trained models."""
        metadata_path = os.path.join(self.models_dir, 'metadata.json')
        preprocessor_path = os.path.join(self.models_dir, 'preprocessing_pipeline.pkl')

        if not os.path.exists(metadata_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                "Model files or metadata not found. Please run 'python train_model.py' first."
            )

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata.get('feature_names', [])

        self.preprocessor = joblib.load(preprocessor_path)

        # Load available models
        model_mappings = {
            'best_model': 'best_model.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'Random Forest': 'random_forest.pkl',
            'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
            'Support Vector Machine': 'support_vector_machine.pkl'
        }

        for name, filename in model_mappings.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
            elif name == 'K-Nearest Neighbors' and os.path.exists(os.path.join(self.models_dir, 'knn.pkl')):
                self.models[name] = joblib.load(os.path.join(self.models_dir, 'knn.pkl'))
            elif name == 'Support Vector Machine' and os.path.exists(os.path.join(self.models_dir, 'svm.pkl')):
                self.models[name] = joblib.load(os.path.join(self.models_dir, 'svm.pkl'))

    def get_available_models(self):
        """Returns list of loaded model names."""
        return [m for m in self.models.keys() if m != 'best_model']

    def validate_input(self, raw_input_dict):
        """
        Validates input fields against dataset schema and numerical bounds.
        Returns: (cleaned_dict, error_messages_list)
        """
        errors = []
        cleaned = {}
        features_meta = self.metadata.get('features', {})

        for feat_name in self.feature_names:
            if feat_name not in raw_input_dict or raw_input_dict[feat_name] is None or raw_input_dict[feat_name] == '':
                errors.append(f"Missing required field: '{features_meta.get(feat_name, {}).get('label', feat_name)}'.")
                continue

            raw_val = raw_input_dict[feat_name]
            meta = features_meta.get(feat_name, {})

            if meta.get('is_numeric', True):
                try:
                    num_val = float(raw_val)
                    if 'int' in meta.get('dtype', ''):
                        num_val = int(round(num_val))
                    
                    # Range check
                    min_val = meta.get('min')
                    max_val = meta.get('max')
                    
                    # Generous bounds allowing reasonable variations beyond min/max
                    if min_val is not None and num_val < (min_val * 0.5 if min_val > 0 else 0):
                        errors.append(f"{meta.get('label', feat_name)} cannot be less than 0.")
                    if max_val is not None and num_val > (max_val * 1.5 if max_val > 0 else 100):
                        errors.append(f"{meta.get('label', feat_name)} value ({num_val}) exceeds acceptable maximum limit.")
                    
                    cleaned[feat_name] = num_val
                except (ValueError, TypeError):
                    errors.append(f"Invalid numeric value for '{meta.get('label', feat_name)}': '{raw_val}'.")
            else:
                cleaned[feat_name] = str(raw_val).strip()

        return cleaned, errors

    def predict_single(self, input_dict, model_name='best_model'):
        """
        Performs end-to-end prediction for a single student.
        """
        cleaned_data, errors = self.validate_input(input_dict)
        if errors:
            return {
                'success': False,
                'errors': errors
            }

        # Resolve selected model
        if model_name not in self.models:
            model_name = 'best_model'
        model = self.models.get(model_name)
        if model is None:
            return {
                'success': False,
                'errors': [f"Selected model '{model_name}' is not available."]
            }

        # Convert to single-row DataFrame with correct column order
        input_df = pd.DataFrame([cleaned_data])[self.feature_names]

        # Transform using preprocessing pipeline
        X_scaled = self.preprocessor.transform(input_df)

        # Predict
        prediction_num = int(model.predict(X_scaled)[0])
        status = 'Placed' if prediction_num == 1 else 'Not Placed'

        # Probabilities & Confidence
        probabilities = None
        confidence = None
        placed_prob = None
        not_placed_prob = None

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[0]
            not_placed_prob = round(float(probs[0]) * 100, 2)
            placed_prob = round(float(probs[1]) * 100, 2)
            confidence = placed_prob if prediction_num == 1 else not_placed_prob
            probabilities = {
                'Placed': placed_prob,
                'Not Placed': not_placed_prob
            }
        else:
            confidence = 85.0  # Fallback default if model lacks predict_proba

        # Generate comparative student insights
        insights = self._generate_student_insights(cleaned_data, status)

        return {
            'success': True,
            'prediction': status,
            'prediction_code': prediction_num,
            'model_used': model_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'student_data': cleaned_data,
            'insights': insights
        }

    def _generate_student_insights(self, cleaned_data, status):
        """Generates analytical feedback comparing student metrics against cohort averages."""
        insights = []
        features_meta = self.metadata.get('features', {})

        for feat, val in cleaned_data.items():
            meta = features_meta.get(feat, {})
            mean = meta.get('mean')
            if mean is not None and meta.get('is_numeric', True):
                diff = val - mean
                is_above = diff >= 0
                pct_diff = round((abs(diff) / mean) * 100, 1) if mean > 0 else 0
                
                # Contextual feedback
                status_type = 'positive' if is_above else 'warning'
                if feat == 'Backlogs':
                    status_type = 'positive' if val == 0 else ('danger' if val > 1 else 'warning')
                    message = f"{val} active backlogs." if val > 0 else "Zero backlogs (Excellent)."
                else:
                    message = f"{val} ({'+' if is_above else '-'}{pct_diff}% vs batch avg {mean})"

                insights.append({
                    'feature': feat,
                    'label': meta.get('label', feat),
                    'value': val,
                    'avg': mean,
                    'status': status_type,
                    'message': message
                })

        return insights


# Global Predictor Singleton instance
_predictor_instance = None

def get_predictor():
    """Returns singleton instance of PlacementPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PlacementPredictor()
    return _predictor_instance


if __name__ == '__main__':
    print("[TEST] Testing PlacementPredictor...")
    predictor = PlacementPredictor()
    print("Available models:", predictor.get_available_models())
    
    # Test High-Probability Sample
    sample_placed = {
        'CGPA': 8.9,
        '10th_Percentage': 85.0,
        '12th_Percentage': 82.0,
        'Internships': 2,
        'Aptitude_Score': 80.0,
        'Communication_Score': 8.5,
        'Technical_Skill_Score': 8.5,
        'Projects': 3,
        'Certifications': 3,
        'Backlogs': 0
    }
    res = predictor.predict_single(sample_placed)
    print("Prediction Result (High Profile):", res['prediction'], "| Confidence:", res['confidence'], "%")

    # Test Borderline/Low Sample
    sample_not_placed = {
        'CGPA': 5.8,
        '10th_Percentage': 55.0,
        '12th_Percentage': 50.0,
        'Internships': 0,
        'Aptitude_Score': 42.0,
        'Communication_Score': 5.2,
        'Technical_Skill_Score': 4.8,
        'Projects': 1,
        'Certifications': 0,
        'Backlogs': 2
    }
    res2 = predictor.predict_single(sample_not_placed)
    print("Prediction Result (Low Profile):", res2['prediction'], "| Confidence:", res2['confidence'], "%")
