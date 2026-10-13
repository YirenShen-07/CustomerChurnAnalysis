# Customer Churn Analysis with Logistic Regression & GAM

An exploratory customer-retention analysis comparing interpretable models of telecom customer churn.

## Business value
Understand relationships between churn and customer tenure, charges, contracts, and services. The analysis can inform retention hypotheses; it does not establish that an intervention reduces churn.

## Technical approach
- Python, Pandas, scikit-learn, pygam, Plotly, and Seaborn.
- Exploratory churn distributions and numeric correlations.
- Removal of customer identifiers, numeric conversion of TotalCharges, median imputation, and one-hot encoding.
- Logistic Regression and Logistic GAM, with a linear-regression exploratory comparison.

## Saved results
| Model | Test accuracy |
| --- | --- |
| Logistic Regression | 81.83% |
| Logistic GAM | 80.70% |

Results are saved notebook outputs from a single 80/20 random split with random_state=42, not a fresh execution or cross-validation estimate. Accuracy alone does not establish improvement over a majority-class baseline.

## Explore and run
Open [the notebook](customer_churn_analysis.ipynb) in Google Colab. Supply the source telecom churn CSV and replace its personal Google Drive path. Install the imported dependencies and run cells in order. A redistributable dataset and a pinned environment are not currently provided.

## Limitations and next steps
- Fit imputers and encoders on training data only.
- Add a majority-class baseline, churn-class precision/recall/F1, ROC-AUC, and a confusion matrix.
- Correct the GAM plot labels to match the actual feature columns: the current labels include Churn even though it is excluded from inputs.
- The GAM uses five spline terms and is not a full-feature equivalent of Logistic Regression.
- Treat associations as hypotheses rather than causal retention recommendations.

## Project context
Developed by Yiren Shen for Duke's AIPI 590 Explainable AI course.
