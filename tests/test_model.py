"""
MindCare AI - Model and Pipeline Unit Tests
=============================================
Tests the machine learning artifact, inference pipeline, feature schema,
unseen category handling, and metrics persistence.
"""

import os
import json
import unittest
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mental_health_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_exists = os.path.exists(MODEL_PATH)
        if cls.model_exists:
            cls.pipeline = joblib.load(MODEL_PATH)
        else:
            cls.pipeline = None

    def test_01_model_artifact_exists(self):
        """Verify the trained model pipeline file exists."""
        self.assertTrue(self.model_exists, f"Model file missing at: {MODEL_PATH}")

    def test_02_metrics_file_validity(self):
        """Verify that model_metrics.json is valid and contains expected metrics."""
        self.assertTrue(os.path.exists(METRICS_PATH), f"Metrics JSON missing at: {METRICS_PATH}")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertIn("performance", metrics)
        self.assertIn("accuracy", metrics["performance"])
        self.assertIn("roc_auc", metrics["performance"])
        self.assertIn("feature_importances", metrics)
        self.assertGreater(metrics["performance"]["accuracy"], 0.70)
        self.assertGreater(metrics["performance"]["roc_auc"], 0.80)

    def test_03_prediction_valid_input(self):
        """Verify inference returns valid binary label and calibrated probability."""
        if not self.pipeline:
            self.skipTest("Pipeline artifact not loaded.")

        sample_input = pd.DataFrame([{
            "Age": 24,
            "Gender": "Female",
            "family_history": "Yes",
            "work_interfere": "Sometimes",
            "benefits": "Yes",
            "care_options": "Yes",
            "anonymity": "Yes",
            "leave": "Somewhat easy"
        }])

        prediction = self.pipeline.predict(sample_input)
        probabilities = self.pipeline.predict_proba(sample_input)

        self.assertIn(prediction[0], [0, 1])
        self.assertEqual(len(probabilities[0]), 2)
        self.assertAlmostEqual(float(probabilities[0][0] + probabilities[0][1]), 1.0, places=5)
        self.assertTrue(0.0 <= probabilities[0][1] <= 1.0)

    def test_04_unseen_and_missing_categories_resilience(self):
        """Verify pipeline gracefully handles unknown categorical values without raising error."""
        if not self.pipeline:
            self.skipTest("Pipeline artifact not loaded.")

        unusual_input = pd.DataFrame([{
            "Age": 35,
            "Gender": "UnknownNovelGender",
            "family_history": "Maybe",
            "work_interfere": "UnseenCategory",
            "benefits": "UnknownVal",
            "care_options": "NotAnOption",
            "anonymity": "NoneOfAbove",
            "leave": "FloatingCategory"
        }])

        # Should execute without throwing ValueError thanks to handle_unknown='ignore'
        preds = self.pipeline.predict(unusual_input)
        probs = self.pipeline.predict_proba(unusual_input)
        self.assertIn(preds[0], [0, 1])
        self.assertEqual(len(probs[0]), 2)

    def test_05_age_boundary_handling(self):
        """Verify pipeline handles lower and upper boundary valid ages."""
        if not self.pipeline:
            self.skipTest("Pipeline artifact not loaded.")

        df_boundary = pd.DataFrame([
            {"Age": 18, "Gender": "Male", "family_history": "No", "work_interfere": "Never", "benefits": "No", "care_options": "No", "anonymity": "No", "leave": "Very easy"},
            {"Age": 70, "Gender": "Female", "family_history": "Yes", "work_interfere": "Often", "benefits": "Yes", "care_options": "Yes", "anonymity": "Yes", "leave": "Very difficult"}
        ])

        preds = self.pipeline.predict(df_boundary)
        self.assertEqual(len(preds), 2)

if __name__ == "__main__":
    unittest.main()

