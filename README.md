# Paisabazaar Banking Fraud Analysis

### Exploratory Data Analysis and Credit Score Classification

A machine-learning internship project completed as part of the
**Labmentix AI/ML Internship**. This project explores customer financial
and credit-behaviour data and builds a classification workflow to
predict the `Credit_Score` category.

> **Scope clarification:** The supplied project brief uses the title
> *Paisabazaar Banking Fraud Analysis*, but the available dataset target
> is `Credit_Score` (`Good`, `Standard`, and `Poor`). The notebook
> therefore focuses on credit-score analysis and classification. It does
> **not** establish or validate a transaction-level fraud-detection
> system.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Business Objective](#business-objective)
-   [Dataset](#dataset)
-   [Project Workflow](#project-workflow)
-   [Exploratory Data Analysis](#exploratory-data-analysis)
-   [Statistical Hypothesis Testing](#statistical-hypothesis-testing)
-   [Machine Learning Workflow](#machine-learning-workflow)
-   [Model Evaluation](#model-evaluation)
-   [Model Persistence and Prediction
    Check](#model-persistence-and-prediction-check)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Installation and Setup](#installation-and-setup)
-   [Running the Notebook](#running-the-notebook)
-   [Key Learnings](#key-learnings)
-   [Limitations and Responsible Use](#limitations-and-responsible-use)
-   [Future Improvements](#future-improvements)
-   [Author](#author)

## Project Overview

Creditworthiness assessment is an important part of lending and
financial risk management. This project examines how financial
attributes and payment behaviour relate to customer credit-score
categories.

The notebook follows a workflow that includes data inspection, data
cleaning, visual exploration, statistical testing, preprocessing,
classification-model training, evaluation, hyperparameter search, and
saving a model for later inference.

This is an educational analysis and prototype. Model results should be
verified by executing the notebook in the intended environment.

## Business Objective

-   Understand the structure and quality of the supplied customer
    dataset.
-   Explore the distribution of credit-score categories.
-   Investigate income, debt, credit utilisation, credit mix, and
    payment delays.
-   Identify correlations and statistical associations.
-   Build and evaluate classifiers for the `Credit_Score` target.
-   Save the selected model and test that it can be loaded to make
    predictions.

## Dataset

**Input file:** `dataset-2.csv`

The notebook is designed around a dataset containing **100,000 rows and
28 columns**. Confirm these dimensions when loading the exact dataset
version used for submission.

The target column is `Credit_Score`, with three categories: `Good`,
`Standard`, and `Poor`.

Examples of fields include:

  Feature                      Description
  ---------------------------- -------------------------------------
  `Age`                        Customer age
  `Annual_Income`              Annual income
  `Monthly_Inhand_Salary`      Monthly in-hand salary
  `Num_Bank_Accounts`          Number of bank accounts
  `Num_Credit_Card`            Number of credit cards
  `Delay_from_due_date`        Payment-delay indicator
  `Num_of_Delayed_Payment`     Number of delayed payments
  `Credit_Mix`                 Credit-mix category
  `Outstanding_Debt`           Outstanding debt
  `Credit_Utilization_Ratio`   Credit-utilisation ratio
  `Credit_History_Age`         Credit-history age field
  `Payment_of_Min_Amount`      Minimum-payment status
  `Credit_Score`               Target class to analyse and predict

Column definitions, units, and data provenance should be verified
against the original data provider where possible. The dataset may
contain repeated customer records, placeholder values, or anomalous
values, so data-quality checks are important.

## Project Workflow

1.  **Load and inspect data** --- examine shape, column names, first and
    last rows, data types, and summary statistics.
2.  **Build a data dictionary** --- document the working meaning of
    important columns.
3.  **Check data quality** --- inspect missing values, duplicates,
    inconsistent categories, placeholders, and unusual values.
4.  **Clean data** --- apply documented cleaning rules without
    overwriting the original CSV.
5.  **Explore distributions** --- examine numerical and categorical
    variables.
6.  **Compare credit-score groups** --- investigate financial attributes
    and payment behaviour across target categories.
7.  **Study correlations and associations** --- use plots and
    statistical tests to investigate relationships.
8.  **Prepare modelling features** --- separate target and input
    variables and exclude identifier-like fields.
9.  **Preprocess and train models** --- use a train/test split and
    preprocessing pipeline.
10. **Evaluate and tune models** --- compare classification metrics and
    test alternative hyperparameters.
11. **Save and reload the selected model** --- use Joblib and run a
    sample prediction check.
12. **Summarise findings and limitations** --- base conclusions on
    executed outputs, not assumptions.

## Exploratory Data Analysis

The notebook includes multiple visualisation types, including: - Count
plots for categorical distributions and credit-score classes. -
Histograms for numerical distributions. - Box plots for comparing
numerical features across credit-score groups. - Bar plots for
categorical comparisons and missing-value summaries where applicable. -
Correlation heatmaps for numerical features. - Pair plots for selected
numerical features and credit-score classes.

Each chart should answer a question about the data: how values are
distributed, whether groups differ, whether outliers are present, or
whether numerical variables move together. Report findings only when
supported by executed chart outputs.

## Statistical Hypothesis Testing

The solution notebook includes three hypothesis-testing questions: 1.
Whether payment-delay distributions differ across credit-score classes.
2. Whether annual-income distributions differ across credit-score
classes. 3. Whether `Credit_Mix` and `Credit_Score` are associated.

The notebook uses tests appropriate to these questions, including the
Kruskal--Wallis test for group-distribution comparisons and the
chi-square test of independence for categorical association.

Interpret p-values in the context of the null hypothesis, test
assumptions, sample size, and chosen significance level. Statistical
significance does not prove causation.

## Machine Learning Workflow

The classification extension uses `Credit_Score` as the target variable.

### Preprocessing

-   Split input features from the target.
-   Exclude direct identifiers and sensitive identifier-like columns
    from predictive features.
-   Impute missing numerical values using the median.
-   Impute missing categorical values using the most frequent category.
-   Scale numerical features where required.
-   One-hot encode categorical features.
-   Fit preprocessing within a pipeline using training data to reduce
    data leakage.

### Models

-   **Logistic Regression** --- baseline linear classifier.
-   **Random Forest** --- ensemble of decision trees.
-   **Extra Trees** --- tree ensemble with additional randomisation.

Hyperparameter search explores alternative configurations. Tuning is not
guaranteed to improve performance.

## Model Evaluation

The notebook evaluates classification models using: - **Accuracy** ---
fraction of correct predictions. - **Precision** --- proportion of
predicted members of a class that are correct. - **Recall** ---
proportion of actual members of a class that the model identifies. -
**F1-score** --- harmonic mean of precision and recall. - **Macro
F1-score** --- equal-weight average of class-level F1-scores. -
**Confusion matrix** --- shows actual classes against predicted classes.

Because the target has multiple categories and may be imbalanced, macro
F1-score and per-class metrics are useful alongside accuracy. Do not add
model scores to this README until the notebook has been run successfully
and the final outputs checked.

## Model Persistence and Prediction Check

The notebook saves the selected estimator in:

``` text
paisabazaar_credit_score_model.joblib
```

It then reloads the model with Joblib and predicts labels for sample
test rows, comparing actual and predicted categories. This is a basic
persistence and inference sanity check, not proof of production
readiness.

Only load Joblib files from trusted sources because serialized Python
objects can execute code when loaded.

## Technology Stack

-   **Language:** Python
-   **Data analysis:** Pandas, NumPy
-   **Visualisation:** Matplotlib, Seaborn
-   **Machine learning:** scikit-learn
-   **Statistical testing:** SciPy
-   **Model persistence:** Joblib
-   **Environment:** Jupyter Notebook / Google Colab / VS Code

## Project Structure

A suggested repository structure is:

``` text
paisabazaar-banking-fraud-analysis/
├── Sample_ML_Submission.ipynb
├── dataset-2.csv                         # Keep private or omit if redistribution is not permitted
├── paisabazaar_credit_score_model.joblib  # Optional generated artifact
├── requirements.txt
└── README.md
```

Actual filenames may differ. Keep the dataset, generated model, and
notebook in the paths expected by your code, or update those paths
before running.

## Installation and Setup

Use Python 3.10 or 3.11 where compatible with your installed package
versions.

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

**Windows:**

``` bash
.venv\Scripts\activate
```

**macOS/Linux:**

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib jupyter ipykernel
```

If the project includes `requirements.txt`, you can instead run:

``` bash
pip install -r requirements.txt
```

## Running the Notebook

Launch Jupyter:

``` bash
jupyter notebook
```

Open the solution notebook and run the cells in order.

Before submitting: - Confirm that the notebook loads the intended CSV
file. - Run all cells from top to bottom without errors. - Review
charts, statistical-test outputs, and evaluation metrics. - Confirm the
model file is saved and can be reloaded. - Ensure conclusions match
actual results. - Include required outputs or supporting files in the
submission folder.

## Key Learnings

-   Inspecting and documenting a tabular dataset.
-   Identifying missing values, inconsistent values, duplicates, and
    outliers.
-   Choosing visualisations for different analytical questions.
-   Interpreting correlation and statistical hypothesis tests.
-   Handling mixed numerical and categorical features.
-   Building reusable preprocessing and modelling pipelines.
-   Comparing multiclass classifiers with appropriate metrics.
-   Understanding why hyperparameter tuning may not improve performance.
-   Saving and reloading a model for inference.
-   Communicating findings, assumptions, and limitations responsibly.

## Limitations and Responsible Use

-   The project brief uses the title *Banking Fraud Analysis*, but the
    available target is a credit-score category. No confirmed fraud
    label or transaction-level fraud field is established by the
    supplied schema.
-   Dataset definitions, units, provenance, and repeated-customer
    structure should be validated.
-   Observational association does not establish causation.
-   Sample predictions do not establish performance on all customers or
    future data.
-   A successful notebook run does not mean the model has been deployed
    to a live server.
-   Real-world credit scoring requires robust validation, fairness and
    bias assessment, security controls, domain review, and human
    oversight.

## Future Improvements

-   Validate data definitions and investigate repeated records at the
    customer level.
-   Add a reproducible experiment log and final evaluation report.
-   Compare tuned and baseline models using the same held-out evaluation
    protocol.
-   Review per-class errors and confusion matrices.
-   Add feature-importance analysis and model explainability where
    appropriate.
-   Evaluate fairness, calibration, and robustness before real-world
    use.
-   If fraud detection is the actual business objective, obtain a
    dataset with verified fraud labels and transaction-level features
    and develop a separate fraud-detection workflow.

## Author

**Abhishek Verma**\
AI/ML Internship Project --- Labmentix

------------------------------------------------------------------------

*This repository is intended for educational and portfolio demonstration
purposes. Verify all reported findings against the executed notebook and
the authorised source dataset.*
