import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('hypertension_data.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classes = np.unique(y_train)
weights = class_weight.compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

#  SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)


#------ Logistic Regression ----------
def train_lr_model(X_train, y_train,strategy_name, C=1.0, max_iter=100, weights='uniform'):
    mlflow.set_experiment("hyper-prediction-v2")

    with mlflow.start_run(run_name= f"LR_{strategy_name}"):
        model = LogisticRegression(C=C, max_iter=max_iter, multi_class='ovr', solver='lbfgs')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
        }

        mlflow.log_params({
            'strategy': strategy_name,
            'C': C,
            'max_iter': max_iter
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:5]
        )
        
        save_lr_plots(model, X_train, y_test, y_pred, y_proba, strategy_name)
        

def save_lr_plots(model, X_train, y_true, y_pred, y_proba, strategy_name):
    
    
    # confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Hyper', 'Hyper'],
               yticklabels=['No Hyper', 'Hyper'])
    plt.title(f'Confusion Matrix ({strategy_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()
    
    #  ROC
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
    
    
# model train              
train_lr_model(X_train, y_train, "Baseline")
train_lr_model(X_train, y_train, "Class_Weights", C=0.5)
train_lr_model(X_train_smote, y_train_smote, "SMOTE", C=2, max_iter=300)

