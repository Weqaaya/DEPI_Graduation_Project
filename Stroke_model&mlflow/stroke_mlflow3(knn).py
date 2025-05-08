import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


df = pd.read_csv('df_cleaned_stroke.csv')
X = df.drop('stroke', axis=1)
y = df['stroke']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
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

def train_knn_model(X_train, y_train, strategy_name, n_neighbors=5, weights='uniform'):
    """دالة لتدريب وتسجيل نموذج KNN"""
    
    mlflow.set_experiment("stroke-prediction-v2")
    
    with mlflow.start_run(run_name=f"KNN_{strategy_name}"):
        # إعداد نموذج KNN
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )
        
        # 
        model.fit(X_train, y_train)
        
        # 
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # calculate matrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # record para
        mlflow.log_params({
            'n_neighbors': n_neighbors,
            'weights': weights,
            'strategy': strategy_name
        })
        
        # record  metrics
        mlflow.log_metrics(metrics)
        
        # model predict
        signature = infer_signature(X_train, model.predict(X_train))
        
        # record model
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name=f"KNN_Stroke_{strategy_name}"
        )
        
        # حفظ الرسومات
        save_knn_plots(y_test, y_pred, y_proba, strategy_name)

def save_knn_plots(y_true, y_pred, y_proba, strategy_name):
    """حفظ رسومات KNN التوضيحية"""
    
    # confusion_matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Stroke', 'Stroke'],
               yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'KNN Confusion Matrix ({strategy_name})')
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
    plt.title(f'KNN ROC Curve ({strategy_name})')
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()

# model para
n_neighbors = 15  
weights = 'distance'  #'uniform'

#  model train
train_knn_model(X_train_scaled, y_train, "Baseline", n_neighbors, weights)
train_knn_model(X_train_scaled, y_train, "Class_Weights", n_neighbors, weights)  # Note: KNN doesn't support class_weight directly
train_knn_model(X_train_smote, y_train_smote, "SMOTE", n_neighbors, weights)