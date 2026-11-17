"""Rebuild fictional training data. No customer records are used."""

import csv
import math
from pathlib import Path
import random


def main():
    rng = random.Random(42)
    path = Path(__file__).resolve().parent / "data/invoices.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["invoice_id", "amount_inr", "payment_terms_days",
                         "previous_late_payments", "paid_late"])
        for index in range(1, 121):
            amount = rng.randint(10, 200) * 500
            terms = rng.choice([15, 30, 45, 60])
            previous = rng.randint(0, 6)
            # Invented educational relationship, not a financial risk formula.
            score = -3.6 + 0.000018 * amount + 0.018 * terms + 0.65 * previous
            probability = 1 / (1 + math.exp(-score))
            paid_late = int(rng.random() < probability)
            writer.writerow([f"INV-{index:03}", amount, terms, previous, paid_late])
    print(f"Created 120 fictional invoices: {path}")


if __name__ == "__main__":
    main()
