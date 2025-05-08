import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix, 
                           roc_curve, auc, precision_score, recall_score,
                           roc_auc_score)  
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.utils import class_weight

# load data
df = pd.read_csv('df_cleaned_stroke.csv')
X = df.drop('stroke', axis=1)
y = df['stroke']

#  split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

# calculate the weights
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

# apply SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

def train_model(X_train, y_train, strategy_name, n_estimators=100, max_depth=10, class_weight=None):
    """دالة لتدريب وتسجيل النموذج"""
    
    mlflow.set_experiment("stroke-prediction-v2")
    
    with mlflow.start_run(run_name=f"RF_{strategy_name}"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)  
        }
        
        mlflow.log_params({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'strategy': strategy_name
        })
        mlflow.log_metrics(metrics)
        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        save_plots(model, X_train, y_test, y_pred, y_proba, strategy_name)

def save_plots(model, X_train, y_true, y_pred, y_proba, strategy_name):
    """حفظ الرسومات التوضيحية"""
    
    # confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Stroke', 'Stroke'],
               yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'Confusion Matrix ({strategy_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()
    
    #  ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({strategy_name})')
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()
    
    # feature_importances
    if len(X_train.columns) <= 30:
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]
        plt.title(f'Feature Importances ({strategy_name})')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

# model parameters
n_estimators = 150
max_depth = 12

# model train
train_model(X_train, y_train, "Baseline", n_estimators, max_depth)
train_model(X_train, y_train, "Class_Weights", n_estimators, max_depth, class_weights)
train_model(X_train_smote, y_train_smote, "SMOTE", n_estimators, max_depth)