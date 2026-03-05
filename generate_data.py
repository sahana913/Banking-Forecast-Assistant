import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# ---------- Loan Default Dataset ----------
def generate_loans(n=1000, seed=42):
    np.random.seed(seed)
    ages = np.random.randint(21, 65, n)
    genders = np.random.choice(["M", "F"], n)
    income = np.random.randint(15000, 200000, n)
    loan_amt = (income * np.random.uniform(3, 8, n)).astype(int)
    tenure = np.random.randint(6, 60, n)
    emi_paid = np.random.randint(1, tenure, n)
    balance = loan_amt - (loan_amt / tenure * emi_paid)
    default_prob = np.clip((loan_amt / (income + 1)) * 0.0008, 0, 1)
    defaulted = (np.random.rand(n) < default_prob).astype(int)

    df_loans = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "age": ages,
        "gender": genders,
        "income": income,
        "loan_amount": loan_amt,
        "tenure_months": tenure,
        "emi_paid": emi_paid,
        "balance": balance.astype(int),
        "defaulted": defaulted
    })
    df_loans.to_csv("data/loans.csv", index=False)
    print(f"✅ loans.csv generated with {n} records.")

# ---------- Liquidity Forecast Dataset ----------
def generate_liquidity(branches=10, days=365, seed=42):
    np.random.seed(seed)
    branch_ids = [f"BR{i:02d}" for i in range(1, branches + 1)]
    start_date = datetime(2024, 1, 1)
    rows = []

    for b in branch_ids:
        inflow = np.random.normal(250000, 40000, days)
        outflow = np.random.normal(230000, 45000, days)
        balance = np.cumsum(inflow - outflow) + 500000
        for i in range(days):
            date = start_date + timedelta(days=i)
            rows.append([b, date.date(), max(inflow[i], 0), max(outflow[i], 0), balance[i]])

    df_liq = pd.DataFrame(rows, columns=["branch_id", "date", "inflow", "outflow", "balance"])
    df_liq.to_csv("data/liquidity.csv", index=False)
    print(f"✅ liquidity.csv generated for {branches} branches × {days} days.")

if __name__ == "__main__":
    generate_loans()
    generate_liquidity()
