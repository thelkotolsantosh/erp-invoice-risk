# ERP Invoice Late-Payment Predictor

A small Python machine learning project for beginners interested in AI, ML, and ERP finance processes.

Give the program a CSV of unpaid invoices. It learns from fictional historical invoices and writes a predicted late-payment score for each new invoice.

## Example

Input: invoice NEW-002, amount INR 85,000, payment terms 60 days, and 6 previous late payments.

Output: a late-payment probability score and either `Likely late` or `Likely on time`. The classification threshold is 0.50.

## Run on Windows

Install Python 3.11 or 3.12. Open PowerShell in the extracted project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

If you installed Python 3.12, change `-3.11` to `-3.12`. No virtual-environment activation is required.

## Run on Linux or macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

The program prints evaluation results and predictions. It also creates `predictions.csv` in your current folder. Running it again replaces that output file.

## Use your own CSV

```powershell
.\.venv\Scripts\python.exe app.py --history data/invoices.csv --input data/new_invoices.csv --output my_predictions.csv
```

Historical and incoming files need these columns:

| Column | Meaning | Example |
| --- | --- | --- |
| invoice_id | Unique invoice identifier within the file | INV-001 |
| amount_inr | Nonnegative invoice amount in Indian rupees | 15000 |
| payment_terms_days | Whole number of days allowed to pay | 30 |
| previous_late_payments | Count of the customer's earlier late payments | 2 |
| paid_late | Historical outcome: 0 = on time, 1 = late; history only | 1 |

Historical data needs at least 20 rows, including at least 5 of each outcome. All feature values must be finite and nonnegative, without missing values. Use past late-payment counts known when each invoice was issued, not counts collected after its outcome.

## How it works

1. Read and validate the historical and new invoice CSVs.
2. Split history into 75% training and 25% testing with a fixed random seed and stratification.
3. Fit StandardScaler and LogisticRegression together in a pipeline using only training rows.
4. Print test accuracy, precision, recall, F1, confusion matrix, and a majority-class baseline.
5. Refit on all historical rows and predict the incoming invoices.
6. Save the invoice details, late probability score, and prediction to CSV.

The pipeline keeps the test rows out of the fitted preprocessing step. See the official [scikit-learn introduction](https://scikit-learn.org/stable/getting_started.html) and [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## Files

| File | Purpose |
| --- | --- |
| app.py | Load CSVs, train, evaluate, and predict |
| requirements.txt | Tested direct Python dependency versions |
| data/invoices.csv | 120 fictional historical invoices |
| data/new_invoices.csv | 3 fictional invoices to predict |
| generate_sample_data.py | Recreate the fictional historical data |
| tests/test_app.py | Verify the prediction flow and input validation |
| .github/workflows/tests.yml | Run tests and the demo on GitHub |
| .gitignore | Keep local environments and output out of Git |
| README.md | Setup and project explanation |
| FAQ.md | Beginner questions and answers |
| LICENSE | MIT license |

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For Linux/macOS, use `.venv/bin/python` instead of the Windows executable path.

## Post on GitHub

1. Sign in to GitHub and click **+ → New repository**.
2. Use the name `erp-invoice-risk` and description `A beginner ML project for predicting invoice late-payment risk from ERP-style CSV data.`
3. Choose visibility. Leave the automatic README, .gitignore, and license options unchecked because these files are included.
4. Click **Create repository**, then **uploading an existing file**.
5. Extract the ZIP. Drag the contents of `erp-invoice-risk` into the upload area, preserving folders. Include `.github` and `.gitignore`; enable hidden-file visibility if needed.
6. Click **Commit changes**. Check the **Actions** tab for the test result.

Alternatively, with Git installed, open a terminal inside the project and run:

```bash
git init
git add .
git commit -m "Add ERP invoice risk ML demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/erp-invoice-risk.git
git push -u origin main
```

Replace `YOUR_USERNAME`. Authenticate with GitHub when Git requests it; never paste an access token into source files or your remote URL.

Suggested repository topics: `python`, `machine-learning`, `erp`, `accounts-receivable`, `scikit-learn`, `beginner-project`.

## Scope and limitations

All sample records are generated using a deliberately invented relationship plus random noise. Scores and test metrics only describe this fictional dataset; they do not demonstrate real-world performance or calibrated business probabilities. The majority baseline gives a simple comparison, not proof of usefulness.

This is an offline learning project using ERP-style data. It does not connect to SAP or Oracle Fusion. No API key, paid service, or GPU is needed. Dependency versions pin direct packages, not every transitive dependency.

For a future real ERP integration, obtain authorized historical data, protect customer details, evaluate on future time periods and separate customers, check calibration and class imbalance, and monitor performance before using predictions in operations.

## License

MIT. See LICENSE.
