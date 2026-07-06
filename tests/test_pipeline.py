"""Unit tests for the attrition pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from train import load_data, get_model


def test_data_exists():
    assert os.path.exists("data/raw/attrition.csv")


def test_data_columns():
    df = pd.read_csv("data/raw/attrition.csv")
    required = [
        "age", "monthly_income", "years_at_company",
        "overtime", "job_satisfaction", "work_life_balance",
        "distance_from_home", "num_companies_worked",
        "training_times_last_year", "performance_rating", "attrition"
    ]
    for col in required:
        assert col in df.columns, f"Missing: {col}"


def test_no_nulls():
    df = pd.read_csv("data/raw/attrition.csv")
    assert df.isnull().sum().sum() == 0


def test_target_is_binary():
    df = pd.read_csv("data/raw/attrition.csv")
    assert set(df["attrition"].unique()) <= {0, 1}


def test_attrition_rate():
    df   = pd.read_csv("data/raw/attrition.csv")
    rate = df["attrition"].mean()
    assert 0.10 <= rate <= 0.25, f"Unexpected rate: {rate:.1%}"


def test_model_builds():
    model = get_model("gradient_boosting", {})
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


def test_model_invalid_raises():
    with pytest.raises(ValueError):
        get_model("not_a_model", {})


def test_model_fits_and_predicts():
    X, y  = load_data()
    model = get_model("logistic", {})
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds) <= {0, 1}
