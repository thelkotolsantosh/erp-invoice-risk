"""Check the data contract and end-to-end prediction output."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from app import ROOT, load_invoices, predict_invoices, train_model


class InvoiceTests(unittest.TestCase):
    def test_training_and_prediction(self):
        history = load_invoices(ROOT / "data/invoices.csv", training=True)
        with contextlib.redirect_stdout(io.StringIO()):
            model = train_model(history)
        incoming = load_invoices(ROOT / "data/new_invoices.csv")
        results = predict_invoices(model, incoming)
        self.assertEqual(results["invoice_id"].tolist(), incoming["invoice_id"].tolist())
        self.assertTrue(results["late_probability"].between(0, 1).all())
        self.assertTrue(results["prediction"].isin(["Likely late", "Likely on time"]).all())

    def test_invalid_features_rejected(self):
        for value in [-1, float("inf"), "not-a-number", None]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "invalid.csv"
                pd.DataFrame([{"invoice_id": "X", "amount_inr": value,
                               "payment_terms_days": 30, "previous_late_payments": 0}]).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_invoices(path)

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.csv"
            path.write_text("invoice_id\nX\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_invoices(path)


if __name__ == "__main__":
    unittest.main()
