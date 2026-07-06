"""
Employee attrition dataset — 1000 employees, ~16% attrition rate.
Mirrors IBM HR Analytics benchmark distribution.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

def make_employees(n, attrited):
    if attrited:
        return pd.DataFrame({
            "age":                      np.random.randint(22, 38, n),
            "monthly_income":           np.random.randint(2000, 6000, n),
            "years_at_company":         np.random.randint(0, 4, n),
            "overtime":                 np.random.choice([1,0], n, p=[0.7,0.3]),
            "job_satisfaction":         np.random.randint(1, 3, n),
            "work_life_balance":        np.random.randint(1, 3, n),
            "distance_from_home":       np.random.randint(10, 30, n),
            "num_companies_worked":     np.random.randint(3, 9, n),
            "training_times_last_year": np.random.randint(0, 2, n),
            "performance_rating":       np.random.randint(1, 3, n),
            "attrition":                [1] * n
        })
    else:
        return pd.DataFrame({
            "age":                      np.random.randint(28, 58, n),
            "monthly_income":           np.random.randint(4000, 20000, n),
            "years_at_company":         np.random.randint(3, 30, n),
            "overtime":                 np.random.choice([0,1], n, p=[0.7,0.3]),
            "job_satisfaction":         np.random.randint(2, 5, n),
            "work_life_balance":        np.random.randint(2, 5, n),
            "distance_from_home":       np.random.randint(1, 15, n),
            "num_companies_worked":     np.random.randint(0, 5, n),
            "training_times_last_year": np.random.randint(2, 6, n),
            "performance_rating":       np.random.randint(3, 5, n),
            "attrition":                [0] * n
        })

n_total    = 1000
n_attrited = int(n_total * 0.16)
n_stayed   = n_total - n_attrited

df = pd.concat([
    make_employees(n_attrited, attrited=True),
    make_employees(n_stayed,   attrited=False)
], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/attrition.csv", index=False)

print(f"Dataset created: {len(df)} employees")
print(f"Attrition rate : {df['attrition'].mean():.1%}")
print(df["attrition"].value_counts().to_string())
