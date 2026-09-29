"""
Automated Test Suite for Student Placement Prediction System
Verifies data loading, model inference, validation, visual assets, and Flask endpoints.
"""

import os
import json
import unittest
import pandas as pd
from app import app
from prediction import PlacementPredictor, get_predictor

class TestPlacementSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.client = app.test_client()

    def test_dataset_and_metadata_integrity(self):
        """Verify dataset and saved metadata match."""
        self.assertTrue(os.path.exists('student_placement.csv'), "student_placement.csv must exist.")
        self.assertTrue(os.path.exists('models/metadata.json'), "Metadata must exist.")
        with open('models/metadata.json', 'r') as f:
            meta = json.load(f)
        self.assertEqual(meta['total_samples'], 1000)
        self.assertEqual(len(meta['feature_names']), 10)
        self.assertIn('CGPA', meta['feature_names'])

    def test_model_files_exist(self):
        """Verify all 5 trained model files and pipelines exist."""
        required_files = [
            'logistic_regression.pkl',
            'decision_tree.pkl',
            'random_forest.pkl',
            'k_nearest_neighbors.pkl',
            'support_vector_machine.pkl',
            'best_model.pkl',
            'preprocessing_pipeline.pkl'
        ]
        for f in required_files:
            path = os.path.join('models', f)
            self.assertTrue(os.path.exists(path), f"Model file '{path}' is missing.")

    def test_high_profile_prediction(self):
        """Test high-performing student prediction."""
        sample = {
            'CGPA': 9.5,
            '10th_Percentage': 90.0,
            '12th_Percentage': 88.0,
            'Internships': 2,
            'Aptitude_Score': 85.0,
            'Communication_Score': 9.2,
            'Technical_Skill_Score': 9.0,
            'Projects': 3,
            'Certifications': 3,
            'Backlogs': 0
        }
        res = self.predictor.predict_single(sample)
        self.assertTrue(res['success'])
        self.assertEqual(res['prediction'], 'Placed')
        self.assertGreater(res['confidence'], 80.0)

    def test_low_profile_prediction(self):
        """Test at-risk student prediction."""
        sample = {
            'CGPA': 5.6,
            '10th_Percentage': 50.0,
            '12th_Percentage': 48.0,
            'Internships': 0,
            'Aptitude_Score': 40.0,
            'Communication_Score': 5.0,
            'Technical_Skill_Score': 4.5,
            'Projects': 1,
            'Certifications': 0,
            'Backlogs': 2
        }
        res = self.predictor.predict_single(sample)
        self.assertTrue(res['success'])
        self.assertEqual(res['prediction'], 'Not Placed')

    def test_invalid_input_validation(self):
        """Verify error handling on incomplete or invalid input."""
        incomplete_sample = {
            'CGPA': 8.5
            # Missing other 9 features
        }
        res = self.predictor.predict_single(incomplete_sample)
        self.assertFalse(res['success'])
        self.assertGreater(len(res['errors']), 0)

    def test_flask_routes(self):
        """Verify all HTML endpoints return HTTP 200."""
        routes = ['/', '/predict', '/comparison', '/visualizations']
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with status {response.status_code}")

    def test_predict_post_endpoint(self):
        """Test HTML Form submission to /predict."""
        form_payload = {
            'selected_model': 'best_model',
            'CGPA': '8.2',
            '10th_Percentage': '80.0',
            '12th_Percentage': '78.0',
            'Internships': '1',
            'Aptitude_Score': '75.0',
            'Communication_Score': '8.0',
            'Technical_Skill_Score': '7.8',
            'Projects': '2',
            'Certifications': '2',
            'Backlogs': '0'
        }
        response = self.client.post('/predict', data=form_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'PLACED', response.data)

    def test_rest_api_endpoints(self):
        """Test REST API /api/predict and /api/metrics."""
        # Metrics API
        metrics_resp = self.client.get('/api/metrics')
        self.assertEqual(metrics_resp.status_code, 200)
        metrics_data = metrics_resp.get_json()
        self.assertTrue(metrics_data['success'])
        self.assertEqual(len(metrics_data['table']), 5)

        # Predict API
        payload = {
            'model': 'best_model',
            'features': {
                'CGPA': 9.0,
                '10th_Percentage': 85.0,
                '12th_Percentage': 80.0,
                'Internships': 1,
                'Aptitude_Score': 80.0,
                'Communication_Score': 8.5,
                'Technical_Skill_Score': 8.5,
                'Projects': 3,
                'Certifications': 2,
                'Backlogs': 0
            }
        }
        api_resp = self.client.post('/api/predict', json=payload)
        self.assertEqual(api_resp.status_code, 200)
        api_data = api_resp.get_json()
        self.assertTrue(api_data['success'])
        self.assertEqual(api_data['prediction'], 'Placed')


if __name__ == '__main__':
    unittest.main()
