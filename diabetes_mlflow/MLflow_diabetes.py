
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    roc_auc_score
)

df = pd.read_csv('/content/cleaned_diabetes_data (3).csv')
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

def save_plots(model, X_train, y_true, y_pred, y_proba, strategy_name):
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'],
               yticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix ({strategy_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({strategy_name})')
    plt.legend(loc='lower right')
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()

    # Feature Importance
    if len(X_train.columns) <= 30:
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({strategy_name})')
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

def train_model(X_train, y_train, X_test, y_test, strategy_name, n_estimators=None, max_depth=None):
    mlflow.set_experiment("diabetes-prediction")

    with mlflow.start_run(run_name=f"RF_{strategy_name}"):
        model = RandomForestClassifier(
            n_estimators=n_estimators if n_estimators else 100,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
        }

        mlflow.log_params({
            'strategy': strategy_name,
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:5])

        save_plots(model, X_train, y_test, y_pred, y_proba, strategy_name)

train_model(X_train, y_train, X_test, y_test, "Original_Default")
train_model(X_train, y_train, X_test, y_test, "RF_150trees_maxdepth12", n_estimators=150, max_depth=12)
train_model(X_train, y_train, X_test, y_test, "RF_300trees_maxdepth20", n_estimators=300, max_depth=20)

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    roc_auc_score
)


df = pd.read_csv('/content/cleaned_diabetes_data (3).csv')
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

def save_plots(model, X_train, y_true, y_pred, y_proba, strategy_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'],
               yticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix ({strategy_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({strategy_name})')
    plt.legend(loc='lower right')
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()

def train_model(strategy_name="LogReg_Default", C=1.0, max_iter=100):
    mlflow.set_experiment("diabetes-logistic")

    with mlflow.start_run(run_name=strategy_name):
        model = LogisticRegression(C=C, max_iter=max_iter, multi_class='ovr', solver='lbfgs')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
        }

        mlflow.log_params({
            'strategy': strategy_name,
            'C': C,
            'max_iter': max_iter
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:5])

        save_plots(model, X_train, y_test, y_pred, y_proba, strategy_name)

train_model("LogReg_Default")
train_model("LogReg_C0.5", C=0.5)
train_model("LogReg_C2_maxiter300", C=2, max_iter=300)

# ========== XGBoost Model ===========
import xgboost as xgb

def train_xgb_model(strategy_name="XGB_Default", n_estimators=100, max_depth=3):
    mlflow.set_experiment("diabetes-xgboost")

    with mlflow.start_run(run_name=strategy_name):
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
        }

        mlflow.log_params({
            'strategy': strategy_name,
            'n_estimators': n_estimators,
            'max_depth': max_depth
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:5])

        save_plots(model, X_train, y_test, y_pred, y_proba, strategy_name)

train_xgb_model("XGB_Default")
train_xgb_model("XGB_150_5", n_estimators=150, max_depth=5)
train_xgb_model("XGB_300_10", n_estimators=300, max_depth=10)



from pyngrok import ngrok

ngrok.set_auth_token("2wokUIqbAWgcDI41DSQT4kgUViQ_4gcvnfQuVQkrEoD5NbLWU")

import mlflow

port = 5000

public_url = ngrok.connect(port)

#!mlflow ui --port {port}

from pyngrok import ngrok
import threading
import os

ngrok.set_auth_token("2wokUIqbAWgcDI41DSQT4kgUViQ_4gcvnfQuVQkrEoD5NbLWU")  

public_url = ngrok.connect(5000)
print("MLflow UI is live at:", public_url)

def run_mlflow():
    os.system("mlflow ui --port 5000")

threading.Thread(target=run_mlflow).start()

