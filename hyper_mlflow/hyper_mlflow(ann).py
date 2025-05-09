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

#------- Artificial Neural Network ------------------

def train_ann_model(X_train, y_train, strategy_name, max_iter,random_state, class_weight=None):
   
    
    mlflow.set_experiment("hyper-prediction-v2")
    
    with mlflow.start_run(run_name=f"ANN_{strategy_name}"):
        ann_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=max_iter,
            random_state=random_state)
        
        ann_model.fit(X_train, y_train)
        
        y_pred = ann_model.predict(X_test)
        y_proba = ann_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)  
        }
        
        mlflow.log_params({
            "hidden_layer_sizes": (100, 50),
            "max_iter" : 500,
            'strategy': strategy_name
        })
        mlflow.log_metrics(metrics)
        
        signature = infer_signature(X_train, ann_model.predict(X_train))
        mlflow.sklearn.log_model(
            ann_model,
            "model",
            signature=signature,
            input_example=X_train[:5]
        )
        
        save_ann_plots(ann_model, X_train, y_test, y_pred, y_proba, strategy_name)

        
def save_ann_plots(model, X_train, y_true, y_pred, y_proba, strategy_name):
    
    
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
    
    
# model parameters

max_iter = 500
random_state = 42
weights = 'distance'

#train model
train_ann_model(X_train, y_train, "Baseline" ,max_iter, random_state,weights)
train_ann_model(X_train, y_train, "Class_Weights",max_iter,random_state,weights)
train_ann_model(X_train_smote, y_train_smote,  "SMOTE",max_iter,random_state,weights)
