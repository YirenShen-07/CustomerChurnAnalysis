"""Reproducible, train-only preprocessing and held-out churn evaluation."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, mean_squared_error
from pygam import LogisticGAM, s, f
from download_data import download

ROOT = Path(__file__).parent
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

def run():
    path = download()
    data = pd.read_csv(path)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    y = data['Churn'].map({'Yes': 1, 'No': 0})
    X = data.drop(columns=['customerID', 'Churn'])
    categorical = [c for c in X if c not in NUMERIC]
    train, test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    preprocess = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), NUMERIC),
        ('categorical', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical)
    ])
    a = preprocess.fit_transform(train)
    b = preprocess.transform(test)
    terms = s(0, n_splines=8) + s(1, n_splines=8) + s(2, n_splines=8) + f(3)
    for i in range(4, a.shape[1]):
        terms += f(i)
    models = {'Majority baseline': DummyClassifier(strategy='most_frequent'),
              'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
              'Logistic GAM': LogisticGAM(terms, lam=.6, max_iter=200)}
    out = ROOT / 'reports'; out.mkdir(exist_ok=True)
    rows = []
    for name, model in models.items():
        model.fit(a, y_train)
        prob = model.predict_proba(b)
        if prob.ndim == 2:
            prob = prob[:, 1]
        pred = (prob >= .5).astype(int)
        rows.append({'Model': name, 'Accuracy': accuracy_score(y_test,pred), 'Churn precision': precision_score(y_test,pred,zero_division=0),
                     'Churn recall': recall_score(y_test,pred,zero_division=0), 'Churn F1': f1_score(y_test,pred,zero_division=0), 'ROC-AUC': roc_auc_score(y_test,prob)})
    metrics = pd.DataFrame(rows).set_index('Model')
    metrics.to_csv(out / 'metrics.csv')
    names = preprocess.get_feature_names_out()
    coefficients = pd.Series(models['Logistic Regression'].coef_[0], index=names)
    coefficients.sort_values().to_csv(out/'logistic_coefficients.csv',header=['Coefficient'])
    fig, ax = plt.subplots(figsize=(10,5))
    metrics[['Accuracy','Churn recall','Churn F1','ROC-AUC']].plot.bar(ax=ax, color=['#244b69','#ea9859','#4f8f83','#8d75a1'])
    ax.set_ylim(0,1);ax.set_ylabel('Held-out score');ax.set_title('Customer churn: accuracy is only part of the story');ax.tick_params(axis='x',rotation=0)
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(5,4))
    pred = models['Logistic Regression'].predict(b)
    ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=['Retained','Churned'],ax=ax,colorbar=False,cmap='Blues')
    ax.set_title('Logistic Regression — held-out customers');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    scale = preprocess.named_transformers_['numeric'].named_steps['standardscaler']
    for i, ax in enumerate(axes):
        grid = models['Logistic GAM'].generate_X_grid(term=i)
        effect, interval = models['Logistic GAM'].partial_dependence(term=i, X=grid, width=.95)
        raw = grid[:,i] * scale.scale_[i] + scale.mean_[i]
        ax.plot(raw,effect,color='#244b69');ax.fill_between(raw,interval[:,0],interval[:,1],alpha=.18)
        ax.set_xlabel(NUMERIC[i]);ax.set_ylabel('Partial log-odds');ax.set_title(NUMERIC[i])
    fig.suptitle('GAM model associations — not causal effects');fig.tight_layout();fig.savefig(out/'gam_effects.png',dpi=160);plt.close(fig)
    linear = LinearRegression().fit(a,y_train)
    metadata = {'seed':42,'split':'stratified 80/20','train_rows':len(train),'test_rows':len(test),'rows':len(data),
                'churn_share':float(y.mean()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'linear_regression_exploratory_mse':float(mean_squared_error(y_test,linear.predict(b))),
                'threshold':.5,'preprocessing_fit':'training split only','GAM_terms':'three numeric splines; categorical indicator factors'}
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(metrics.round(4).to_string())
    return metrics

if __name__ == '__main__':
    run()
