# Frequently Asked Questions

1. **What problem does this solve?** It demonstrates predicting whether an invoice will be paid after its due date.
2. **Where is the AI?** Logistic regression learns statistical relationships from labeled historical invoices.
3. **What does ERP mean?** Enterprise Resource Planning: software for business activities such as finance, purchasing, and inventory.
4. **Does it connect to Oracle Fusion or SAP?** No. It reads local ERP-style CSV files.
5. **Why this project?** It combines a small ML workflow with a familiar accounts-receivable process.
6. **Is the sample data real?** No. Every record and outcome is fictional.
7. **What is a feature?** An input: amount, payment terms, or previous late-payment count.
8. **What is the target?** `paid_late`, the historical outcome the model learns to predict.
9. **Why logistic regression?** It is a compact baseline classifier that provides probability scores.
10. **Why pandas?** It reads CSV files and handles tabular data.
11. **Why scale inputs?** Amounts and counts use different scales; StandardScaler standardizes them before model fitting.
12. **Why a pipeline?** It fits preprocessing and the classifier together, using only the training portion during evaluation.
13. **Why split the data?** Held-out rows let us inspect predictions on records the model did not train on.
14. **What is the baseline?** A simple classifier that always predicts the most frequent training label.
15. **What does 0.50 mean?** Scores at or above 0.50 are classified as likely late. This is an illustrative threshold.
16. **Can I trust the score for business decisions?** No. The fictional-data demo does not establish real-world accuracy or probability calibration.
17. **Why these dependency versions?** They match the versions used to run and verify this project. They are not claimed to be the latest releases.
18. **Do I need an API key or GPU?** No. The program runs locally on a CPU.
19. **Why does it train every time?** It keeps this small demo simple and avoids distributing a serialized model. Larger projects usually separate training and inference.
20. **What should I add next?** A web interface, authorized ERP exports, time-based evaluation, and customer-level validation.
21. **What do precision and recall mean here?** Late-payment precision measures how many invoices flagged as late were actually late. Late-payment recall measures how many actual late invoices the model caught. Both matter when reviewing false alarms and missed payments.
22. **What does the confusion matrix show?** Rows are actual outcomes and columns are predicted outcomes, ordered as on time, then late. The off-diagonal cells count incorrect predictions.
23. **Why use random_state=42?** A fixed seed makes the train/test split repeatable. The number 42 has no special statistical meaning, and results can differ across software versions.
24. **What inputs should I avoid?** Do not include future information such as the current invoice's payment date or days overdue after settlement. Features must be available at the time you intend to predict.
25. **What files should stay out of GitHub?** Keep customer data, passwords, API tokens, local virtual environments, and generated prediction exports out of the repository. The included sample CSVs contain fictional records and are safe to share.
