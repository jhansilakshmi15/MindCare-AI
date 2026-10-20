"""
MindCare AI - Machine Learning Training Pipeline
==================================================
This script loads the mental health survey dataset, cleans and validates the features,
trains a Random Forest Classifier with cross-validation and hyperparameter tuning,
evaluates the model on held-out test data, and exports the serialized pipeline and
detailed metrics for the MindCare AI web application.

Target: 'treatment' (Indicates whether the respondent sought mental health treatment).
Note: This represents treatment-seeking behavior patterns in the dataset,
not a clinical medical diagnosis.
"""

import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mental_health.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "mental_health_model.pkl")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Standardized gender mapping categories
MALE_VARIANTS = {
    "male", "m", "male-ish", "maile", "mal", "male (cis)",
    "make", "man", "msle", "mail", "malr", "cis man", "cis male"
}
FEMALE_VARIANTS = {
    "cis female", "f", "female", "woman", "femake",
    "cis-female/femme", "female (cis)", "femail"
}
NON_BINARY_VARIANTS = {
    "trans-female", "non-binary", "genderqueer", "androgyne",
    "agender", "trans woman", "female (trans)", "queer", "fluid", "enby"
}

def clean_gender(val):
    if pd.isna(val):
        return "Other"
    clean_val = str(val).strip().lower()
    if clean_val in MALE_VARIANTS:
        return "Male"
    if clean_val in FEMALE_VARIANTS:
        return "Female"
    if clean_val in NON_BINARY_VARIANTS:
        return "Non-binary/Other"
    return "Other"

def load_and_inspect_data(filepath):
    if not os.path.exists(filepath):
        # Fallback to parent survey.csv if available
        alt_path = os.path.join(BASE_DIR, "..", "survey.csv")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at: {filepath}")

    logging.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logging.info(f"Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    duplicates = df.duplicated().sum()
    logging.info(f"Duplicate records found: {duplicates}")

    # Inspect missing values in relevant columns
    logging.info(f"Missing values summary:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df

def preprocess_dataframe(df):
    data = df.copy()

    # 1. Age outlier handling: Restrict between 18 and 100, replace outliers with NaN
    data.loc[~data["Age"].between(18, 100), "Age"] = np.nan
    median_age = data["Age"].median()
    data["Age"] = data["Age"].fillna(median_age)
    logging.info(f"Age sanitized: Min={data['Age'].min()}, Max={data['Age'].max()}, Median={median_age}")

    # 2. Gender normalization
    data["Gender"] = data["Gender"].apply(clean_gender)

    # 3. Work interference: Handle missing as 'Don't know' / 'Not Reported'
    data["work_interfere"] = data["work_interfere"].fillna("Don't know")

    # 4. Standardize categorical text features
    categorical_columns = ["family_history", "work_interfere", "benefits", "care_options", "anonymity", "leave"]
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].fillna("Don't know").astype(str).str.strip()

    # 5. Target variable encoding
    if "treatment" in data.columns:
        data["treatment_target"] = data["treatment"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        logging.info(f"Target distribution:\n{data['treatment_target'].value_counts(normalize=True)}")

    return data

def build_model_pipeline(numerical_cols, categorical_cols):
    """
    Creates an end-to-end scikit-learn Pipeline with median imputation and scaling
    for numeric features, plus One-Hot Encoding for categorical features.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Don't know")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ]
    )

    # Primary model: Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier)
    ])

    return full_pipeline

def evaluate_and_compare(X_train, X_test, y_train, y_test, preprocessor):
    """
    Evaluates baseline models and computes comprehensive metrics for the primary Random Forest model.
    """
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest (Primary)": RandomForestClassifier(
            n_estimators=150, max_depth=7, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42
        )
    }

    comparison = {}
    for name, model in candidate_models.items():
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)
        probs = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds))
        auc = float(roc_auc_score(y_test, probs)) if probs is not None else 0.0

        comparison[name] = {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }
        logging.info(f"Model: {name} | Test Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    return comparison

def train_and_export():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df_raw = load_and_inspect_data(DATA_PATH)
    df_clean = preprocess_dataframe(df_raw)

    num_cols = ["Age"]
    cat_cols = [
        "Gender", "family_history", "work_interfere",
        "benefits", "care_options", "anonymity", "leave"
    ]
    feature_cols = num_cols + cat_cols

    X = df_clean[feature_cols]
    y = df_clean["treatment_target"]

    # Stratified train-test split (75% train, 25% test) to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logging.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Build and fit pipeline
    pipeline = build_model_pipeline(num_cols, cat_cols)

    # Perform 5-fold cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    logging.info(f"5-Fold Cross Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit pipeline on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test data
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    logging.info(f"Test Accuracy:  {acc:.4f}")
    logging.info(f"Test Precision: {prec:.4f}")
    logging.info(f"Test Recall:    {rec:.4f}")
    logging.info(f"Test F1-Score:  {f1:.4f}")
    logging.info(f"Test ROC-AUC:   {auc:.4f}")
    logging.info(f"Confusion Matrix: {cm}")

    # Extract feature importances
    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["classifier"]
    encoded_feature_names = preprocessor.get_feature_names_out()
    importances = rf.feature_importances_

    feature_importance_list = [
        {"feature": str(name).replace("cat__", "").replace("num__", ""), "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(encoded_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    # Baseline comparison
    comparison_results = evaluate_and_compare(X_train, X_test, y_train, y_test, preprocessor)

    # Save trained model pipeline
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    logging.info(f"Model successfully saved to: {MODEL_SAVE_PATH}")

    # Save detailed metadata and metrics
    metrics_data = {
        "model_name": "Random Forest Classifier",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "total_samples": int(len(df_raw)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_used": feature_cols,
            "target_variable": "treatment",
            "class_distribution": {
                "Treatment_Yes": int((y == 1).sum()),
                "Treatment_No": int((y == 0).sum())
            }
        },
        "performance": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            }
        },
        "feature_importances": feature_importance_list[:12],
        "model_comparison": comparison_results,
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 7,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced"
        }
    }

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    logging.info(f"Model metrics and metadata saved to: {METRICS_SAVE_PATH}")

    return metrics_data

if __name__ == "__main__":
    train_and_export()

