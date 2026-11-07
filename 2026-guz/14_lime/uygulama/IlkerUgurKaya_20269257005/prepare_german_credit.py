#!/usr/bin/env python3
"""UCI german.data dosyasını Kaggle'daki okunabilir 9 özellik + Risk yapısına çevirir."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


UCI_COLUMNS = [
    "checking_code", "Duration", "credit_history_code", "purpose_code",
    "Credit amount", "savings_code", "employment_code", "installment_rate",
    "personal_status_sex_code", "other_debtors_code", "residence_since",
    "property_code", "Age", "other_installment_code", "housing_code",
    "existing_credits", "job_code", "dependents", "telephone_code",
    "foreign_worker_code", "risk_code",
]

SEX = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}
JOB = {"A171": 0, "A172": 1, "A173": 2, "A174": 3}
HOUSING = {"A151": "rent", "A152": "own", "A153": "free"}
SAVINGS = {"A61": "little", "A62": "moderate", "A63": "quite rich", "A64": "rich", "A65": "unknown/no account"}
CHECKING = {"A11": "little", "A12": "moderate", "A13": "rich", "A14": "no account"}
PURPOSE = {
    "A40": "car", "A41": "car", "A42": "furniture/equipment",
    "A43": "radio/TV", "A44": "domestic appliances", "A45": "repairs",
    "A46": "education", "A47": "vacation/others", "A48": "education",
    "A49": "business", "A410": "vacation/others",
}
RISK = {1: "good", 2: "bad"}


def convert(input_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path, sep=r"\s+", header=None, names=UCI_COLUMNS)
    if raw.shape != (1000, 21):
        raise ValueError(f"1000 satır ve 21 alan bekleniyordu; bulunan: {raw.shape}")

    clean = pd.DataFrame({
        "Age": raw["Age"],
        "Sex": raw["personal_status_sex_code"].map(SEX),
        "Job": raw["job_code"].map(JOB),
        "Housing": raw["housing_code"].map(HOUSING),
        "Saving accounts": raw["savings_code"].map(SAVINGS),
        "Checking account": raw["checking_code"].map(CHECKING),
        "Credit amount": raw["Credit amount"],
        "Duration": raw["Duration"],
        "Purpose": raw["purpose_code"].map(PURPOSE),
        "Risk": raw["risk_code"].map(RISK),
    })
    if clean.isna().any().any():
        raise ValueError("Bilinmeyen bir UCI kodu bulundu; eşleme tablosunu kontrol edin.")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="UCI german.data yolu")
    parser.add_argument("-o", "--output", type=Path, default=Path("german_credit_clean.csv"))
    args = parser.parse_args()
    clean = convert(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    print(f"{len(clean)} satır yazıldı: {args.output}")
    print(clean["Risk"].value_counts().to_string())


if __name__ == "__main__":
    main()
