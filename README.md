# Venture Funding Classification — Baselines and Neural Network

This project compares classifiers on a historical applicant dataset. It includes a preserved **unfinished TensorFlow course notebook** and a separate **runnable scikit-learn implementation** with a majority-class baseline, logistic regression, and a small multilayer perceptron (MLP) neural network.

The objective is to demonstrate feature preparation, correct train/test separation, reproducible evaluation, and limitations—not to make real financial eligibility decisions.

## Important finding in the historical notebook

The [original TensorFlow exercise](Starter_Code/venture_funding_with_deep_learning.ipynb) includes unfinished `# YOUR CODE HERE` code cells and cannot run end to end as checked in. It also includes `IS_SUCCESSFUL` (the prediction target) in its list of categorical input columns, creating **target-leakage risk**. Its one-hot encoding step may also remove the unencoded target column that later cells expect.

This cleanup **does not claim to finish that notebook** or to validate a TensorFlow model. It preserves the historical artifact and adds an independently testable implementation.

## Run the reference pipeline

Install Python 3.11 or 3.12 and create an isolated environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

From the repository root:

```bash
python venture_funding.py --model all --output reports/latest.json
```

For faster evaluation of a single model:

```bash
python venture_funding.py --model logistic
python venture_funding.py --model mlp --sample 1500 --output reports/sample-mlp.json
```

The script loads [the original historical CSV](Starter_Code/Resources/applicants_data.csv). It validates binary target labels and handles both categorical and numeric features. It drops `EIN`, `NAME`, and **`IS_SUCCESSFUL`** from model inputs before training.

Every model uses the same stratified holdout split (25% by default, random seed 1). The preprocessing pipeline learns imputation values, category encodings, and scaling parameters **from the training portion only**. Test records are never used to fit preprocessing.

## Results and evaluation

The JSON report includes separate holdout accuracy, balanced accuracy, precision, recall, and F1 for each model. It records the split sizes and feature names, so reviewers can inspect whether target and identity fields were accidentally included.

A **majority-class dummy baseline** helps detect misleading accuracy in imbalanced classification datasets. Logistic regression provides an interpretable linear baseline; `MLPClassifier` demonstrates a simple multilayer neural network using scikit-learn.

This pipeline is **not** TensorFlow/PyTorch, and it is not production-ready. It does not implement model deployment, independent fairness assessment, cross-validation, or calibrated decision thresholds.

## Verify

```bash
python -m unittest discover -s tests -v
```

GitHub Actions tests the code on Python 3.11 and 3.12, executes a full historical-data baseline, and runs a neural-network smoke test on a deterministic sample. These checks are separate from the lightweight source-integrity workflow.

## Portfolio relevance and ethics

The project demonstrates supervised learning, categorical encoding, leakage prevention, class imbalance awareness, repeatable splits, and reproducible metrics—useful skills for machine-learning internship interviews.

The data is educational and historical; its provenance and licensing have not been independently established in this cleanup. No predictions should be used for actual loans, grants, hiring, or funding decisions. Any production model would require proper permission to use the data, robust validation, monitoring, and fairness review.
