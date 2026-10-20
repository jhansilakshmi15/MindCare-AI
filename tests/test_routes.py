"""
MindCare AI - Route and Web Application Integration Tests
===========================================================
Tests all Flask endpoints, form input validation, error handling,
consent-based storage, guest sessions, and health checks.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
import database

class TestAppRoutes(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()

    def test_01_home_page(self):
        """Verify home landing page loads with HTTP 200 and key headings."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Understand Your Wellbeing", response.data)
        self.assertIn(b"Start Screening", response.data)

    def test_02_questionnaire_page(self):
        """Verify questionnaire page renders successfully."""
        response = self.client.get("/questionnaire")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Screening Questionnaire", response.data)
        self.assertIn(b"Demographics", response.data)

    def test_03_wellness_resources_page(self):
        """Verify wellness resources page renders with verified helplines."""
        response = self.client.get("/resources")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Wellness Support", response.data)
        self.assertIn(b"14416", response.data) # Tele-MANAS helpline
        self.assertIn(b"KIRAN", response.data)

    def test_04_user_dashboard_page(self):
        """Verify user dashboard page loads."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Personal Screening Dashboard", response.data)

    def test_05_admin_page(self):
        """Verify administrator & model insights page loads with metrics."""
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model Insights", response.data)
        self.assertIn(b"Random Forest Feature Importance", response.data)

    def test_06_health_check_endpoint(self):
        """Verify /health returns JSON with healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_07_valid_prediction_submission(self):
        """Verify valid form submission redirects or renders result page with score."""
        payload = {
            "age": "22",
            "gender": "Female",
            "family_history": "Yes",
            "work_interfere": "Sometimes",
            "leave": "Somewhat difficult",
            "care_options": "No",
            "benefits": "No",
            "anonymity": "Don't know",
            "stress_level": "7",
            "sleep_quality": "Fair",
            "workload": "Heavy",
            "consent_saved": "1"
        }

        response = self.client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Screening Results", response.data)
        self.assertIn(b"Recommended Next Steps", response.data)
        self.assertIn(b"Submitted Responses Summary", response.data)

    def test_08_invalid_age_submission(self):
        """Verify negative or out-of-range age values are rejected cleanly."""
        payload = {
            "age": "150", # Beyond 100 max
            "gender": "Male",
            "family_history": "No"
        }
        response = self.client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Should redirect back to questionnaire and flash error
        self.assertIn(b"Age must be between 18 and 100", response.data)

    def test_09_non_numeric_age_submission(self):
        """Verify alphabetic age string is rejected."""
        payload = {
            "age": "twenty",
            "gender": "Male"
        }
        response = self.client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Age must be a valid whole number", response.data)

    def test_10_missing_age_submission(self):
        """Verify empty age is rejected."""
        payload = {
            "age": "",
            "gender": "Male"
        }
        response = self.client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter your age", response.data)

    def test_11_404_error_page(self):
        """Verify non-existent route renders custom 404 page."""
        response = self.client.get("/non-existent-page-url")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Page Not Found (404)", response.data)

    def test_12_delete_history(self):
        """Verify delete-history clears session records."""
        response = self.client.post("/api/delete-history", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Successfully deleted", response.data)

if __name__ == "__main__":
    unittest.main()
