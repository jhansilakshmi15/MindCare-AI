"""
MindCare AI - Web Application Backend (Flask)
==============================================
A full-stack mental health risk screening and wellness support system.
Designed for educational screening demonstration, machine learning evaluation,
and student viva presentation.

Core Principles:
- Strictly educational: Never claims to diagnose or replace clinical consultation.
- Privacy by design: Anonymous screening supported; consent required to store records.
- Robust validation: All inputs validated against dataset-supported schemas.
- Single model loading at startup with graceful fallback.
"""

import os
import json
import uuid
import logging
import joblib
import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, abort
)

import database

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mental_health_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "mindcare-ai-secure-btech-secret-key-2026")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Global model and metrics cache
ML_PIPELINE = None
MODEL_METRICS = {}

def load_ml_artifacts():
    """Loads the trained ML pipeline and metrics once during application initialization."""
    global ML_PIPELINE, MODEL_METRICS
    if os.path.exists(MODEL_PATH):
        try:
            ML_PIPELINE = joblib.load(MODEL_PATH)
            logging.info("Trained machine learning pipeline loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model pipeline: {e}")
            ML_PIPELINE = None
    else:
        logging.warning(f"Model file not found at {MODEL_PATH}. Run 'python train_model.py' first.")
        ML_PIPELINE = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                MODEL_METRICS = json.load(f)
            logging.info("Model metrics loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load metrics JSON: {e}")
            MODEL_METRICS = {}
    else:
        MODEL_METRICS = {}

# Initialize database tables and artifacts
database.init_db()
load_ml_artifacts()

@app.before_request
def ensure_session_id():
    """Assigns an anonymous UUID to guest sessions for local tracking."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def clean_gender_input(val):
    if not val:
        return "Other"
    clean_val = str(val).strip().lower()
    if clean_val in ["male", "m", "man"]:
        return "Male"
    if clean_val in ["female", "f", "woman"]:
        return "Female"
    if clean_val in ["non-binary", "transgender", "other", "prefer not to say"]:
        return "Non-binary/Other"
    return "Other"

def categorize_risk(prob):
    """
    Categorizes the estimated probability into calibrated, non-stigmatizing tiers.
    """
    if prob >= 0.65:
        return {
            "tier": "Elevated Screening Indicator",
            "badge_class": "badge-elevated",
            "color": "#E11D48",
            "summary": "Responses align with patterns of individuals experiencing significant workplace or academic strain who benefited from professional support.",
            "action_header": "Consider Professional Evaluation & Care",
            "recommendation": "We strongly recommend scheduling an appointment with a licensed counselor, therapist, or university mental health service for a comprehensive assessment."
        }
    elif prob >= 0.40:
        return {
            "tier": "Moderate Screening Indicator",
            "badge_class": "badge-moderate",
            "color": "#D97706",
            "summary": "Responses suggest mild-to-moderate strain or mixed institutional/wellness support factors.",
            "action_header": "Focus on Proactive Stress Reduction & Dialogue",
            "recommendation": "Incorporate active stress management techniques, review academic/work boundaries, and consider speaking with a peer counselor or wellness mentor."
        }
    else:
        return {
            "tier": "Low Screening Indicator",
            "badge_class": "badge-low",
            "color": "#059669",
            "summary": "Responses suggest manageable stress levels with adequate supportive factors currently in place.",
            "action_header": "Maintain Healthy Wellbeing Habits",
            "recommendation": "Continue positive sleep hygiene, physical activity, and social connections. Revisit the screening periodically if circumstances change."
        }

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------
@app.route("/")
def index():
    """Home landing page."""
    return render_template("index.html")

@app.route("/questionnaire")
def questionnaire():
    """Multi-step questionnaire interface."""
    return render_template("questionnaire.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Validates form inputs, passes verified features to the ML model,
    saves the record if consent was granted, and displays results.
    """
    global ML_PIPELINE
    if ML_PIPELINE is None:
        # Attempt to reload if user trained model after server started
        load_ml_artifacts()
        if ML_PIPELINE is None:
            return render_template("error.html",
                error_title="Model Not Ready",
                error_message="The machine learning model artifact was not found. Please run 'python train_model.py' to generate 'models/mental_health_model.pkl'.",
                show_train_hint=True
            ), 503

    try:
        # 1. Parse and validate Age
        raw_age = request.form.get("age", "").strip()
        if not raw_age:
            flash("Please enter your age.", "danger")
            return redirect(url_for("questionnaire"))
        try:
            age = int(raw_age)
            if age < 18 or age > 100:
                flash("Age must be between 18 and 100 for this screening.", "danger")
                return redirect(url_for("questionnaire"))
        except ValueError:
            flash("Age must be a valid whole number.", "danger")
            return redirect(url_for("questionnaire"))

        # 2. Extract and validate model features
        gender_raw = request.form.get("gender", "Prefer not to say").strip()
        gender = clean_gender_input(gender_raw)

        family_history = request.form.get("family_history", "No").strip()
        if family_history not in ["Yes", "No"]:
            family_history = "No"

        work_interfere = request.form.get("work_interfere", "Don't know").strip()
        valid_work_interfere = ["Never", "Rarely", "Sometimes", "Often", "Don't know"]
        if work_interfere not in valid_work_interfere:
            work_interfere = "Don't know"

        benefits = request.form.get("benefits", "Don't know").strip()
        if benefits not in ["Yes", "No", "Don't know"]:
            benefits = "Don't know"

        care_options = request.form.get("care_options", "Not sure").strip()
        if care_options not in ["Yes", "No", "Not sure"]:
            care_options = "Not sure"

        anonymity = request.form.get("anonymity", "Don't know").strip()
        if anonymity not in ["Yes", "No", "Don't know"]:
            anonymity = "Don't know"

        leave = request.form.get("leave", "Don't know").strip()
        valid_leave = ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]
        if leave not in valid_leave:
            leave = "Don't know"

        # 3. Optional Holistic Lifestyle Self-Check (Kept separate from ML features)
        raw_stress = request.form.get("stress_level", "5")
        try:
            stress_level = max(1, min(10, int(raw_stress)))
        except ValueError:
            stress_level = 5

        sleep_quality = request.form.get("sleep_quality", "Fair")
        workload = request.form.get("workload", "Manageable")
        consent_saved = bool(request.form.get("consent_saved"))

        # 4. Construct feature DataFrame strictly matching the trained pipeline schema
        feature_dict = {
            "Age": [age],
            "Gender": [gender],
            "family_history": [family_history],
            "work_interfere": [work_interfere],
            "benefits": [benefits],
            "care_options": [care_options],
            "anonymity": [anonymity],
            "leave": [leave]
        }
        input_df = pd.DataFrame(feature_dict)

        # 5. Execute ML prediction
        prediction_label = int(ML_PIPELINE.predict(input_df)[0])
        probabilities = ML_PIPELINE.predict_proba(input_df)[0]
        # Probability of class 1 (treatment / elevated indicator)
        est_prob = float(probabilities[1])

        risk_meta = categorize_risk(est_prob)

        # 6. Prepare summary record
        record = {
            "user_id": session.get("user_id"),
            "session_id": session.get("session_id"),
            "age": age,
            "gender": gender_raw,
            "family_history": family_history,
            "work_interfere": work_interfere,
            "benefits": benefits,
            "care_options": care_options,
            "anonymity": anonymity,
            "leave": leave,
            "stress_level": stress_level,
            "sleep_quality": sleep_quality,
            "workload": workload,
            "prediction_label": prediction_label,
            "prediction_prob": round(est_prob, 3),
            "risk_tier": risk_meta["tier"],
            "consent_saved": consent_saved
        }

        # 7. Persist to database if user gave explicit consent
        record_id = None
        if consent_saved:
            record_id = database.save_screening(record)

        # 8. Save latest result in session for review/printing
        session["latest_result"] = {
            "record": record,
            "risk_meta": risk_meta,
            "prob_percentage": round(est_prob * 100, 1),
            "record_id": record_id
        }

        return redirect(url_for("result"))

    except Exception as e:
        logging.exception("Error processing prediction request:")
        return render_template("error.html",
            error_title="Inference Error",
            error_message=f"An unexpected error occurred while processing the questionnaire: {str(e)}"
        ), 500

@app.route("/result")
def result():
    """Displays the result of the most recent screening."""
    res_data = session.get("latest_result")
    if not res_data:
        flash("No active screening result found. Please complete the questionnaire first.", "info")
        return redirect(url_for("questionnaire"))

    return render_template("result.html",
        record=res_data["record"],
        risk_meta=res_data["risk_meta"],
        prob_percentage=res_data["prob_percentage"],
        record_id=res_data.get("record_id")
    )

@app.route("/resources")
def resources():
    """Wellness recommendations and verified emergency support contacts."""
    return render_template("resources.html")

@app.route("/dashboard")
def dashboard():
    """User history dashboard showing trend of past screenings."""
    user_id = session.get("user_id")
    session_id = session.get("session_id")

    records = database.get_screenings_for_user_or_session(user_id=user_id, session_id=session_id)

    # Prepare timeline data for Chart.js
    dates = [r["created_at"][:16] for r in records]
    scores = [round(r["prediction_prob"] * 100, 1) for r in records]
    stress_scores = [r["stress_level"] * 10 for r in records] # scaled to 0-100 for comparison

    latest_record = records[-1] if records else None

    return render_template("dashboard.html",
        records=records,
        total_count=len(records),
        latest_record=latest_record,
        chart_dates=json.dumps(dates),
        chart_scores=json.dumps(scores),
        chart_stress=json.dumps(stress_scores)
    )

@app.route("/admin")
def admin():
    """
    Demonstration administrator and model insights dashboard.
    Visualizes model metrics, feature importances, and aggregate statistics.
    """
    global MODEL_METRICS
    if not MODEL_METRICS and os.path.exists(METRICS_PATH):
        load_ml_artifacts()

    db_stats = database.get_aggregate_admin_stats()

    # Prepare Chart.js arrays for feature importance
    top_features = MODEL_METRICS.get("feature_importances", [])[:8]
    fi_labels = [f["feature"] for f in top_features]
    fi_values = [f["importance"] for f in top_features]

    return render_template("admin.html",
        metrics=MODEL_METRICS,
        db_stats=db_stats,
        fi_labels=json.dumps(fi_labels),
        fi_values=json.dumps(fi_values)
    )

@app.route("/api/delete-history", methods=["POST"])
def delete_history():
    """Deletes all saved screenings for the current user or guest session."""
    user_id = session.get("user_id")
    session_id = session.get("session_id")
    deleted = database.delete_screenings(user_id=user_id, session_id=session_id)
    if "latest_result" in session:
        session.pop("latest_result", None)

    flash(f"Successfully deleted {deleted} saved screening record(s).", "success")
    return redirect(url_for("dashboard"))

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": ML_PIPELINE is not None,
        "database_connected": os.path.exists(database.DB_PATH),
        "version": "1.0.0"
    })

# -------------------------------------------------------------------------
# Optional Authentication Routes
# -------------------------------------------------------------------------
@app.route("/auth/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = database.verify_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/auth/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user_id, err = database.create_user(username, password)
        if err:
            flash(err, "danger")
        else:
            session["user_id"] = user_id
            session["username"] = username
            flash("Account registered successfully! You are now logged in.", "success")
            return redirect(url_for("dashboard"))
    return render_template("register.html")

@app.route("/auth/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# -------------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html",
        error_title="Page Not Found (404)",
        error_message="The requested URL could not be found. Please return to the homepage."
    ), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html",
        error_title="Internal Server Error (500)",
        error_message="An unexpected server error occurred. Please try again later."
    ), 500

if __name__ == "__main__":
    import socket
    port = int(os.environ.get("PORT", 5000))
    # Test if preferred port is available, otherwise switch to 5001
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.close()
    except OSError:
        port = 5001
    
    print(f" * MindCare AI Server online at: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
